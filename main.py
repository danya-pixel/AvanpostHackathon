import streamlit as st

from ml.finetuner import finetune_model
from tasks.objects import DATASET_PATH

st.title("ResNet 50 trainer")

# if __name__ == '__main__':
#     result = finetune_model(data_dir=DATASET_PATH,
#                             classes_names=["trucks", "minibus", "ski", "dump_trucks", "bicycles", "snowboard",
#                                            "tractor", "trains", "gazon", "horses"],
#                             pth_path="tmp/9cf19667-8f1b-4000-ae25-fef7826f3579/model/model.pth",
#                             new_data_dir="tmp/9cf19667-8f1b-4000-ae25-fef7826f3579/images",
#                             new_data_name="скейтборд"
#                             )
