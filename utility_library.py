import numpy as np
import csv
import requests
import os
import openai
import asyncio
import aiofiles

# This is a test procedure that validates classification capabalities
# across various machine learning models more specifically Large Language models
# Author: Kevin McNamara
# Initital Date of Release 20.May.2023
# Subsequent Edits

## read data from a REST API
def get_data_from_rest_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error: HTTP status code {}.".format(response.status_code))
        return None

## CSV writer
def write_data_to_csv(data, filename):
    try:
        with open(filename, "a") as f:
            f.write(data + "\n")
    except FileNotFoundError:
        with open(filename, "w") as f:
            f.write("")
        with open(filename, "a") as f:
            f.write(data + "\n")

## CSV Reader
def read_csv_file(filename):
    with open(filename, "r", newline="") as f:
        return list(csv.reader(f))

##convert a list to an array
def list_to_string_array(lst):
    return np.array([str(element) for element in lst])

## prepend to an array per row
def prepend_text_to_strings(strings, text):
    return [text + s for s in strings]

## converts a json object to a string for a csv row
def json_to_csv_row(json_object):
    return " ".join(str(value) for key, value in json_object.items())

##split string and read n column
def split_string_and_read_column(string,pattern,column):
    return string.split(pattern)[column]

## read data from a REST API
def get_data_from_rest_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error: HTTP status code {}.".format(response.status_code))
        return None

## remove escape characters from a string
def remove_escape_characters(input):
    characters = ["'", "\n", "\t", "\r", "\b", "\f", "\"", "[", "]", "\\"]
    for char in characters:
        input = input.replace(char, "")
    return input

## converts a json object to a string for a csv row
def json_to_csv_row(json_object):
    return " ".join(str(value) for key, value in json_object.items())

##split string and read n column
def split_string_and_read_column(string,pattern,column):
    return string.split(pattern)[column]

#OpenAI Specific Section
async def get_classifier(data):
    response = openai.Completion.create(
        model="text-davinci-003",
        #engine="gpt-3.5-turbo",  # GPT-3.5 Turbo engine
        prompt= data,
        temperature=0.5,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response

def format_record_from_openai(response, row):
    raw = split_string_and_read_column(json_to_csv_row(response)," JSON: ",1)
    #raw = split_string_and_read_column(remove_escape_characters(raw.replace(" ", "").strip()),"}{",0)
    raw = split_string_and_read_column(remove_escape_characters(raw),"}{",0)
    raw = split_string_and_read_column(raw,"text:",1)
    raw = remove_escape_characters(row) + ","+raw
    raw = raw.replace(" : ", "\",\"")
    raw = "\"" + raw + "\""
    raw = raw.replace(", nn", "\",\"")
    raw = raw.replace("}", "\"")
    raw = split_string_and_read_column(raw,"{",0)
    return raw

async def call_openai_api_v1(clean, fileToWriteTo):
    recordLine = 0
    ## target openai rest interface
    url = "https://api.example.com/v1/data"
    ##openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = "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"
    ## loop through rows and write results to file
    for row in clean:
        recordLine +=1
        response = await get_classifier(remove_escape_characters(row))
        record = format_record_from_openai(response, row)
        write_data_to_csv(record,fileToWriteTo)
        print(f"Row: '{recordLine}' "+record)
        #await asyncio.sleep(1)

async def call_openai_api_v1_protected_concurrent_threads(clean, fileToWriteTo):
    url = "https://api.example.com/v1/data"
    openai.api_key = "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"

    # Define semaphore to limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(5)

    async def process_row(sem, row, fileToWriteTo):
        # Acquire semaphore
        async with sem:
            response = await get_classifier(remove_escape_characters(row))
            record = format_record_from_openai(response, row)

            # Open the file in append mode and write record
            async with aiofiles.open(fileToWriteTo, "a") as f:
                await f.write(record)
                await f.write('\n')

            print(f"Row: '{row}' "+record)

    # Create tasks for each row
    tasks = [process_row(semaphore, row, fileToWriteTo) for row in clean]

    # Run tasks concurrently
    await asyncio.gather(*tasks)

async def call_openai_api_v1_protected_concurrent_threads_with_retry(numberSemiphores, maxRetries, clean, fileToWriteTo):
    url = "https://api.example.com/v1/data"
    openai.api_key = "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"
    max_retries = maxRetries  # Set maximum number of retries

    # Define semaphore to limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(numberSemiphores)

    async def process_row(sem, row, fileToWriteTo):
        # Acquire semaphore
        async with sem:
            for attempt in range(max_retries):
                try:
                    response = await get_classifier(remove_escape_characters(row))
                    record = format_record_from_openai(response, row)
                    break  # If the function call was successful, break from the loop
                except Exception as e:
                    if attempt < max_retries - 1:  # If we're not at the last attempt
                        print(f"Failed to process row '{row}', attempt {attempt+1}. Retrying...")
                        await asyncio.sleep(1)  # Optionally sleep before retrying
                        continue
                    else:  # If we're at the last attempt, re-raise the exception
                        raise e from None

            # Open the file in append mode and write record
            async with aiofiles.open(fileToWriteTo, "a") as f:
                await f.write(record)
                await f.write('\n')

            print(f"Row: '{row}' "+record)

    # Create tasks for each row
    tasks = [process_row(semaphore, row, fileToWriteTo) for row in clean]

    # Run tasks concurrently
    await asyncio.gather(*tasks)
