import openai
import json


openai.api_key = "key"

# create the training data to create a fine-tuned model for DMV


# CLI data preparation tool
# Data must be in JSONL document
# CLI: openai tools fine_tunes.prepare_data -f <LOCAL_FILE>

# convert json to jsonl
# prompt has to end with the indicator string `? ->
# completion with a suffix \n.

# read the json file
with open("training_data.json", "r") as input_file:
    training_data = json.load(input_file)

file_name = "fineTune_data.jsonl"
with open(file_name, "w") as output_file:
    for entry in training_data:
        json.dump(entry, output_file)
        output_file.write("\n")


# Upload training data to OpenAI to use the file id to fine tune the model
upload_response = openai.File.create(
    file=open(file_name, "rb"),
    purpose='fine-tune'
)

file_id = upload_response.id
print(file_id)


# Create a fine-tuned model
# CLI openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>

fine_tuning = openai.FineTune.create(training_file=file_id, model="davinci")
# print(fine_tuning)


# List all events
# CLI: openai api fine_tunes.list
# fine_tune_events = openai.FineTune.list_events(id=fine_tuning.id)
# print(fine_tune_events)


# retrieve fine tune job
# CLI: openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>
# export OPENAI_API_KEY= < key > set env variable
# openai api fine_tunes.follow -i < key >
retrieve_response = openai.FineTune.retrieve(id=fine_tuning.id)
print(retrieve_response)

# When job status is "succeeded", you can use the fine_tuned model
