import asyncio
from utility_library import *

# This is a test procedure that validates classification capabalities
# across various machine learning models more specifically Large Language models
# Author: Kevin McNamara
# Initital Date of Release 20.May.2023
# Subsequent Edits
# usage: provide input CSV file + prompt per record + output file name

testFileName = "./testdata.csv"
engineeredPrompt = "label for sentiment  : "
outputFileName = "./PII.csv"
numberSemaphores = 10
maxRetries =5

## Main
async def main():
    ## read in test data
    testData = list_to_string_array(read_csv_file(testFileName))

    ## embed engineered prompts
    preparedData = prepend_text_to_strings(testData,engineeredPrompt)

    ## call api and write results to local file OPENAI
    await call_openai_api_v1_protected_concurrent_threads_with_retry(numberSemaphores, maxRetries, preparedData, outputFileName)

if __name__ == "__main__":
    asyncio.run(main())
