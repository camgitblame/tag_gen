The requirements.txt file was re-built as we added new modules to the code to make it easier to revert back to a safe version.
The requirements_genre_rag.txt should contain all the necessary packages for all of the models.
The training command to train each model is located in the corresponding run.sh scripts.
With the training completed and the gpt2-output folder(s) created, the corresponding inference.py file can be run.
Each inference file contains a field for MODEL_DIR and EVAL_FILE. 
MODEL_DIR should be gpt2-output for normal models, gpt2-output-genre-boosted for files, and these folders are created after training.
EVAL_FILE should be eval_genre.csv for genre models and eval.csv for normal models, as the genre csvs has an extra column.
Each inference file generates a "generated_vs..." csv that shows a side by side comparison of the generated taglines with the original taglines.
