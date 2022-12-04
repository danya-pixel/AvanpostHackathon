import streamlit as st
import pandas as pd
from celery.result import AsyncResult
from utils.upload_to_fds import upload
from tasks.predict import predict_by_model
from tasks.objects import celery

st.title("Finetune")

st.text("Здесь вы можете дообучить модель на еще один класс")


def clear_state():
    if 'task_id' in st.session_state:
        if st.session_state['task_id'] is not None:
            celery.control.revoke(st.session_state['task_id'], terminate=True)
        del st.session_state['task_id']
    if 'async_result' in st.session_state:
        del st.session_state['async_result']


st.button("Начать заново", on_click=clear_state)


def form_upload():
    with st.form("fine_tune"):
        model_archive = st.file_uploader("Archive with model (.zip)", type="zip")
        class_name = st.text_input("New class name")
        submitted = st.form_submit_button("train")
        if submitted:
            if model_archive is None or class_name is None:
                st.write("You shoud provide all files")
                return
            clear_state()
            st.write("Супер! Отправляю в космос....")
            model_url = upload(model_archive)
            st.write("Ваши файлы загружены")
            st.write(f"Model: {model_url}")
            st.write(f"Class: {class_name}")
            task = predict_by_model.apply_async(args=(images_url, model_url))
            st.write(f"Task id: {task.task_id}")
            return task.task_id


if 'task_id' not in st.session_state:
    task_id = form_upload()
    if task_id is not None:
        st.session_state['task_id'] = task_id
else:
    task_id = st.session_state['task_id']


def get_results():
    task_id = st.session_state['task_id']
    async_result = AsyncResult(id=task_id, app=celery)
    st.session_state['async_result'] = async_result


def rerun():
    st.session_state['async_result'].retry(countdown=2, max_retries=1)


if task_id is not None:
    with st.container():
        st.button("Проверить результаты", on_click=get_results)
        if 'async_result' in st.session_state:
            res = st.session_state['async_result']
            st.text(f"Статус задачи {res.task_id}: {res.status}")
            if res.status == "SUCCESS":
                result = res.get()
                st.text(f"Results:")
                df = pd.DataFrame(result.items(), columns=["Имя файла", "Класс"]).sort_values(
                    by="Имя файла").reset_index(drop=True)
                st.dataframe(df)
            elif res.status == "FAILURE":
                st.button("Перезапустить", on_click=get_results)
                st.error(f"Error: {res.get()}")
