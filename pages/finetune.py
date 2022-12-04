import streamlit as st
import pandas as pd
import math
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from celery.result import AsyncResult
from utils.upload_to_fds import upload
from tasks import finetune_model, celery

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
            # model_url = 'https://fds.es.nsu.ru/uploads/888fdd38-1659-4577-b4ba-ddafc0cedae9'
            st.write("Ваши файлы загружены")
            st.write(f"Model: {model_url}")
            st.write(f"Class: {class_name}")
            task = finetune_model.apply_async(args=(class_name, model_url))
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
                col1, col2 = st.columns(2)
                result, classes, _ = res.get()
                with col1:
                    st.text("Классы модели:")
                    for i in result['classes']:
                        st.text(f"* {i}")

                with col2:
                    st.text("Метрики:")
                    st.metric("F1", value=round(result['f1'], 4))
                    st.metric("Accuracy", value=round(result['acc'], 4))
                    st.text("Обучение:")
                    x = range(len(result['epoch_loss']))
                    loss = result['epoch_loss']
                    acc = result['epoch_acc']
                    fig = plt.figure(figsize=(5, 5))
                    fig, axs = plt.subplots(nrows=1, ncols=2)
                    axs[0].set_title("Accuracy")
                    axs[0].set_xlabel("Epoch")
                    axs[0].plot(x, acc)
                    axs[1].set_xlabel("Epoch")
                    axs[1].set_title("Loss")
                    axs[1].plot(x, loss)
                    st.pyplot(fig)

                # st.text("Loss: ")
                # loss_graph = figure(
                #     title='simple line example',
                #     x_axis_label='epoch',
                #     y_axis_label='Loss')
                # loss_graph.line(x, loss, legend_label='Loss', line_width=2, color="red")
                # # p.line(x, acc, legend_label='Accuracy', line_width=2, color="green")
                # # df = pd.DataFrame(result.items(), columns=["Имя файла", "Класс"]).sort_values(
                # #     by="Имя файла").reset_index(drop=True)
                # # st.dataframe(df)
                # # st.bokeh_chart(p, use_container_width=True)
                # st.text("Accuracy: ")
                # acc_graph = figure(
                #     title='simple line example',
                #     x_axis_label='epoch',
                #     y_axis_label='Loss')
                # acc_graph.line(x, acc, legend_label='Accuracy', line_width=2, color="green")
                # st.bokeh_chart(loss_graph | acc_graph, use_container_width=True)
            elif res.status == "FAILURE":
                st.button("Перезапустить", on_click=get_results)
                st.error(f"Error: {res.get()}")
