## submit job on server

**use screen:**

* screen -ls
* screen -S run_name
* screen -r 1234  # resume screen with id 1234
* detach screen from terminal Ctrl + a + d
* screen -X -S 1234 quit  # kill screen


## mlflow

#### copy run-data to local file system:
```bash
scp -r rpeer@ameisenbaer.cosy.sbg.ac.at:/home/rpeer/masters_project/graph_conv_net/experiments/mlruns .
```

##### graphical user interface

```bash
mlflow server
```
show runs: http://127.0.0.1:5000

#### download csv file with all runs of an experiment
```bash
mlflow experiments csv -x 1 -o experiment_runs.csv
```

with python API:  
https://databricks.com/blog/2019/10/03/analyzing-your-mlflow-data-with-dataframes.html

## LaTex

fix bibliography:
```
bibtex Thesis.aux
``` 
If no new Thesis.bbl and Thesis.blg are created, create them manually using this command.