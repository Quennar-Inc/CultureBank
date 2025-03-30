# Step By Step Help

## Head to the WorkBench
Visit [Google Cloud Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench)

1. Select the CultureBank Instance and click on the Start on top nav
2. Once Started, click on the JupyterLab button in front of the instance
3. Open a terminal where you can run git commands (directly connect to our git hub repositories, or run programs, install dependencies, etc)

## Helpful Commands
- `cd ..` - Go back to the previous directory in the terminal if needed
- `ls` - List the files in the current directory
- `pwd` - Print the current working directory
- `clear` - Clear the terminal (ctrl + l also works)
- You can click on folder icon on the left to navigate to the folder and see the files/folders from left panel

## Setup Instructions

### Navigate to CultureBank Directory
In a terminal, ensure you're in CultureBank directory. You should see something like:
```bash
(base) jupyter@culture-bank-instace:~/CultureBank$
```

If needed, navigate to the folder:
```bash
cd CultureBank
# cd .. to go back to the previous directory in the terminal if needed
```

### Dependencies (Already Installed)
> **Note**: The following steps are already completed in this environment and can be ignored.

```bash
# install dependencies
pip install -r requirements.txt

# create conda environment and activate it
conda create -n culturebank 

# activate the environment
conda activate culturebank

# OpenAI API key is already added to the environment variable
# export OPENAI_API_KEY="your_openai_api_key_goes_here" # Replace with your actual OpenAI API key
```

## Running the Pipeline

### Test Pipeline (Step 0)
To test if the pipeline is working, run the following command (make sure you're in the CultureBank directory):
```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python data_process_pipeline/main.py -i 0 -c data_process_pipeline/configs/config_dummy_data_vanilla_mistral.yaml
```

> **Note**: If `CultureBank/data_process_pipeline/results` folder is not empty, running the programs will prompt you to overwrite the files. Type `Y` or `YES` to overwrite the files.

### Running Pipeline Step by Step
You can run individual steps by changing the step number (0 is for step 0, 1 is for step 1, etc):
```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python data_process_pipeline/main.py -i 0 -c data_process_pipeline/configs/config_dummy_data_vanilla_mistral.yaml
```

### Running Multiple Steps
To run steps 0,1,3,4,5,6,7,8:
```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python data_process_pipeline/main.py -i 0,1,3,4,5,6,7,8 -c data_process_pipeline/configs/config_dummy_data_vanilla_mistral.yaml
```

> **Important**: We need to set up GPUs to run the entire pipeline (current environment is CPU only, GPUs will be added soon). 