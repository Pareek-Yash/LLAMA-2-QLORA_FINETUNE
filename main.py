from training_qlora import load_model_for_training, train_model
from data_processing import load_data_from_json
if __name__=="__main__":
    model_name = "tiiuae/falcon-7b-instruct"
    training = False

    if training == True:
        model, tokenizer = load_model_for_training(model_name=model_name)

        OUTPUT_DIR = "models/"

        json_file_path = "data/input.json"
        data = load_data_from_json(json_file_path=json_file_path)

        train_model(
            model=model,
            tokenizer=tokenizer,
            data=data,
            OUTPUT_DIR=OUTPUT_DIR
        )