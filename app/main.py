import base64
import io
import secrets
from pathlib import Path
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import edge_tts
from uuid import uuid4
from app.config import settings
from app.agent import graph

ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title='Medical Agent Educational API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins.split(','),allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Authorization','Content-Type'])
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')


class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=4000)
    history: list[dict[str,str]] = Field(default_factory=list,max_length=10)
    image_data: str | None = None
    image: str | None = None

def auth(authorization: str | None):
    if settings.api_token and not secrets.compare_digest(authorization or '',f'Bearer {settings.api_token}'):
        raise HTTPException(status_code=401,detail='Invalid API token')


def inspect_image(data):
    if not data:
        return ''
    try:
        header,payload=data.split(',',1)
        mime=header.split(';')[0].split(':')[1]
        raw=base64.b64decode(payload,validate=True)
        if mime not in ('image/jpeg','image/png','image/webp') or len(raw)>5_000_000:
            raise ValueError('Only JPEG/PNG/WebP under 5MB supported')
    except (ValueError,IndexError) as exc:
        raise HTTPException(status_code=400,detail=f'Invalid image: {exc}')

    try:
        client = genai.Client(api_key=settings.effective_gemini_api_key)
        result = client.models.generate_content(
            model=settings.gemini_vision_model,
            contents=[
                types.Part.from_bytes(data=raw, mime_type=mime),
                'Describe only objectively visible medical-relevant features. Do not diagnose, estimate severity, or assert absence of illness. If image unclear, say so.'
            ]
        )
        return result.text or ''
    except Exception as exc:
        print('Vision error:', repr(exc), flush=True)
        return ''


@app.get('/',response_class=HTMLResponse)
def home():
    return (ROOT/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {'status':'ok','configured':bool(settings.effective_gemini_api_key and settings.pinecone_api_key),'tavily':bool(settings.tavily_api_key),'provider':'gemini'}



@app.post('/chat')
@app.post('/api/chat')
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias='X-Conversation-ID')
):
    auth(authorization)

    if not settings.effective_gemini_api_key or not settings.pinecone_api_key:
        raise HTTPException(503,'Set GEMINI_API_KEY and PINECONE_API_KEY in .env')

    history=[x for x in payload.history if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)][:10]

    # SQLite persistent memory: reuse the same conversation thread
    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        raw_img = payload.image_data or payload.image
        observation=inspect_image(raw_img)

        state=graph.invoke(
            {
                'question':payload.message,
                'history':history,
                'image_description':observation,
                'kb':[],
                'web':[],
                'answer':'',
                'route':''
            },
            config=config
        )

        sources = []
        for x in state.get('kb', []):
            sources.append({
                'type': 'pdf',
                'label': f"Gale Encyclopedia of Medicine (2002) - Page {x.get('page')}",
                'page': x.get('page'),
                'year': 2002,
                'score': x.get('score'),
                'excerpt': x.get('text', '')
            })
        for x in state.get('web', []):
            sources.append({
                'type': 'web',
                'label': x.get('title', 'Clinical Web Reference'),
                'title': x.get('title', ''),
                'url': x.get('url', ''),
                'year': 2024,
                'excerpt': x.get('content', '')
            })

        return {
            'answer':state['answer'],
            'sources': sources,
            'tools': ['pinecone']+(['tavily'] if state['route']=='web' and settings.tavily_api_key else [])+(['vision'] if observation else []),
            'conversation_id':thread_id
        }

    except HTTPException:
        raise
    except Exception as exc:
        print('Chat error:',repr(exc),flush=True)
        raise HTTPException(502,'AI service failed. Check server logs, credentials, Pinecone index, and connectivity.')


@app.post('/api/transcribe')
async def transcribe(file:UploadFile=File(...),authorization:str|None=Header(default=None)):
    auth(authorization)
    data=await file.read(10_000_001)
    if len(data)>10_000_000:
        raise HTTPException(413,'Audio exceeds 10 MB')
    try:
        client = genai.Client(api_key=settings.effective_gemini_api_key)
        result = client.models.generate_content(
            model=settings.gemini_vision_model,
            contents=[
                types.Part.from_bytes(data=data, mime_type=file.content_type or 'audio/webm'),
                'Please transcribe this medical question audio accurately. Output only the verbatim transcription text without any additional commentary.'
            ]
        )
        return {'text': (result.text or '').strip()}
    except Exception as exc:
        print('Transcribe error:', repr(exc), flush=True)
        raise HTTPException(502,'Speech transcription unavailable')

@app.post('/api/speak')
async def speak(payload:dict,authorization:str|None=Header(default=None)):
    auth(authorization)
    text=str(payload.get('text',''))[:3000]
    if not text:
        raise HTTPException(400,'Missing text')
    try:
        communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        audio_stream.seek(0)
        return Response(content=audio_stream.read(),media_type='audio/mpeg')
    except Exception as exc:
        print('Speak error:', repr(exc), flush=True)
        raise HTTPException(502,'Speech synthesis unavailable')