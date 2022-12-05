# AvanpostHackathon

# How to run
Поднимаем redis
```bash 
$ docker-compose up -d 
```
Поднимаем воркера на машине с gpu
```bash
$ export FDS_JWT_TOKEN="<TOKEN_FROM_FDS>"
$ export CELERY_RESULT_BACKEND="redis://<IP:PORT for redis>/0"
$ export CELERY_BROKER_URL="redis://<IP:PORT for redis>/0"
$ celery -A tasks.celery  worker -l info 
```

Поднимаем фронтенд 
```bash
$ export FDS_JWT_TOKEN="<TOKEN_FROM_FDS>"
$ export CELERY_RESULT_BACKEND="redis://<IP:PORT for redis>/0"
$ export CELERY_BROKER_URL="redis://<IP:PORT for redis>/0"
$ streamlit run main.py
```
