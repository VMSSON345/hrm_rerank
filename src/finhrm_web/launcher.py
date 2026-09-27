"""Local gateway and post-HRM contract runtime; no legacy backend switch."""
from __future__ import annotations
import argparse,json,os,subprocess,threading,time,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def ensure_mineru(url):
    try:
        with urllib.request.urlopen(url.rstrip('/')+'/openapi.json',timeout=2):return
    except OSError:pass
    parsed=urllib.parse.urlsplit(url)
    if parsed.hostname not in {'127.0.0.1','localhost'}:return
    executable=ROOT/'.venv-mineru/bin/mineru-api'
    if not executable.is_file():
        print('MinerU API chưa có; các Markdown đã chuẩn bị vẫn dùng được.',flush=True);return
    logs=ROOT/'tmp/finhrm_web';logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.setdefault('MINERU_MODEL_SOURCE','local');env.setdefault('OMP_NUM_THREADS','4');env.setdefault('MKL_NUM_THREADS','4')
    with (logs/'mineru_contract.log').open('ab') as log:
        process=subprocess.Popen([str(executable),'--host','127.0.0.1','--port',str(parsed.port or 8000)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (logs/'mineru_contract.pid').write_text(str(process.pid),encoding='utf8')
    print('MinerU khởi động riêng; nhật ký: tmp/finhrm_web/mineru_contract.log',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8765')))
    parser.add_argument('--model',default=os.environ.get('MODEL_PATH',str(ROOT/'models/document_hrm_V14_1_bkai_scratch_seed31_v1.pt')))
    parser.add_argument('--mineru-api-url',default=os.environ.get('MINERU_API_URL','http://127.0.0.1:8000'))
    parser.add_argument('--no-mineru-start',action='store_true')
    parser.add_argument('--database',default=os.environ.get('FINHRM_WEB_DB_PATH',str(ROOT/'tmp/finhrm_web/workspace.sqlite3')))
    args=parser.parse_args();os.chdir(ROOT)
    os.environ['FINHRM_SERVICE_URL']=f'http://127.0.0.1:{args.port+1}'
    os.environ['FINHRM_WEB_DB_PATH']=args.database
    if not args.no_mineru_start:ensure_mineru(args.mineru_api_url)
    import torch
    torch.manual_seed(31);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_service import PersistentDocumentHRMService,DocumentIngestionManager,make_handler,_ServiceHTTPServer
    from hrm_finance_qa.document_sources import OCRConfig
    from finhrm_web.app import create_app
    import uvicorn
    checkpoint=Path(args.model).resolve();bundle=DocumentHRMBundle.load(checkpoint,device='cpu')
    ingestion=DocumentIngestionManager(index_path=ROOT/'cache/document_page_index.sqlite3',ocr_config=OCRConfig(enabled=True,mineru_api_url=args.mineru_api_url,mineru_cache_dir=ROOT/'cache/mineru',mineru_command=ROOT/'.venv-mineru/bin/mineru'),allowed_roots=[ROOT])
    service=PersistentDocumentHRMService(bundle,ingestion,model_path=checkpoint,device='cpu')
    server=_ServiceHTTPServer(('127.0.0.1',args.port+1),make_handler(service,upload_root=ROOT/'tmp/demo_uploads'))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    print(f'FinHRM http://127.0.0.1:{args.port} | validation_engine={service.validation_backend} | legacy_symbolic_active=False',flush=True)
    try:uvicorn.run(create_app(),host='127.0.0.1',port=args.port,log_level='info')
    finally:server.shutdown();server.server_close();ingestion.shutdown()
if __name__=='__main__':main()
