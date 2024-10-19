# Getting started with DVC

## Initializing DVC

1. **Initialize DVC:**

```bash
dvc init
```

Initializing DVC will create several files where metadata for data artifacts (data, models, metrics) will be stored.

## Versioning Data
1. **Add the dataset to DVC tracking:**

```bash
dvc add data.csv
```

2. **Stage the changes in Git:**

```bash
git add data.csv.dvc .gitignore
```

3. **Commit the changes:**

```bash
git commit -m "Adding raw data"
```

### Setting Up Remote Storage

Before setting up remote storage, ensure that the necessary dependencies are installed if you’re using a cloud provider, in addition to having programmatic access credentials and installing the library that allows you to connect to the cloud provider.

```bash
pip install 'dvc[gdrive]'
pip install 'dvc[s3]'
pip install 'dvc[gcs]'
pip install 'dvc[azure]'
```

#### Example with AWS

### Install the library

```bash
pip install awscli
```

### Enter configuration credentials (AWS Access Key Id, AWS Secret Access Key)

```bash
aws configure
```


1. **Example of configuring Google Drive as remote storage:**

```bash
dvc remote add -d remote-storage gdrive://{Google Drive folder ID}
```

**Note**
Currently, there are issues with authentication on Google’s side. If authentication with Google Drive fails, you may consider the following options:


1. **Example of configuring a local folder as remote storage:**

```bash
dvc remote add -d local_remote /Users/your_username/Documents/Demo1/local_storage
```

1. **Example of an Amazon S3 bucket as remote storage:**

```bash
dvc remote add -d s3_storage s3://{bucket_name}/{optional_folder}
```

2. **Commit the remote storage configuration:**

```bash
git add .dvc/config
git commit -m "Setting up remote"
```

3. **Upload data to remote storage:**

```bash
dvc push
```

### Fetching Data from Remote Storage

We will simulate fetching data from local storage.

1. **You can delete the cache and the dataset:**

```bash
rm -rf wine_quality_df.csv
rm -rf .dvc/cache
```

2. **Retrieve the data from remote storage:**

```bash
dvc pull
```

### Modifying the File and Tracking Changes

For the dataset we are working with, we can remove records using a text editor, delete the header, or copy and paste more rows. Once the changes are made, we add them to DVC.

1. **Add changes to DVC:**

```bash
dvc add wine_quality_df.csv
```

2. **Stage the changes in Git:**

```bash
git add wine_quality_df.csv.dvc
```

3. **Commit the changes:**

```bash
git commit -m "Removing lines"
```

4. **Push the changes to remote storage:**

```bash
dvc push
```

**Note**
Every change made to data artifacts, when pushed to remote storage, will create a new folder where the metadata for those changes will be stored.

### Retrieve a Previous Version of the Dataset

To work with a previous version of the dataset and fetch it from remote storage:

1. **Checkout the previous version of the `.dvc` file:**

```bash
git checkout HEAD^1 wine_quality_df.csv.dvc
```

2. **Update the local data to match the version we just checked out:**

```bash
dvc checkout
```

3. **Commit the changes:**

```bash
git commit -m "Reverting changes"
```

## ML pipeline development
You can find several py files inside the folder `mlops`. Those files are the steps for the ML pipeline, each file contains the necessary functions and work as independent files also. The steps are:
1. Load data
2. Preprocess data and split
3. Train model
4. Evaluate model

In order to build the pipeline, DVC must be initialized. Files will be created and a push to remote repository is necessary. The file called `dvc.yaml` contains the stages and the respective order, dependencies and outputs so we can build the pipeline.

Once the stages are set, we should run the following command to execute the stages sequentially:
```bash
dvc repro
```

If you execute the following command, tou will see the steps that are included in your pipeline and where defined in the dvc.yaml file.
```bash
dvc dag
```

## Integrating DVC with MLflow

There are lines in the py file that will track your experiments on the MLflow UI, everytime you run `dvc_repro`.

As well, keys in params.yaml and the dvc.yaml are included for the option of running not only the current model, but another one listed as well.

Remember that *mlflow* must be installed and once you have installed, it should be initialized from a terminal.

```bash
pip install mlflow
```

Then, for initializing *mlflow*

```bash
mlflow ui
```

After this, you now would be able to run mlflow on the IP **http://127.0.0.1:5000**, then, you are ready to run your pipeline with `dvc repro`.