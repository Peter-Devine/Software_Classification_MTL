import random
from evaluation_engines import create_eval_engine
from model_saver import ModelSaver
import time

def train_on_tasks(task_dict, PARAMS, logger, is_fine_tuning):
    model_saver = ModelSaver(model_dir="./models")

    # Get the per task eval metric against which best models are chosen
    task_eval_metrics = {task_name: [0] for task_name, task in task_dict.items()}

    # Evaluation engine for each task
    task_eval_engines = {task_name: create_eval_engine(model=task.model, is_multilabel=task.is_multilabel, n_classes=task.n_classes, cpu=PARAMS.cpu) for task_name, task in task_dict.items()}

    # Get a list of task names to determine order of training, with one entry for each batch of that task (e.g. [Maalej2015, Maalej2015, Maalej2015] for Maalej2015 if it had 3 batches)
    task_training_list = []
    for task_name, task in task_dict.items():
        task_training_list.extend([task_name]*task.train_length)

    random.seed(PARAMS.random_state)

    if not is_fine_tuning:
        # Shuffle task list during multi-task training so that tasks are trained roughly evenly throughout
        random.shuffle(task_training_list)

    # initialize global step number
    step_num = 0
    # Record the number of steps taken for each task in a dict
    task_steps = {task_name: 0 for task_name, task in task_dict.items()}
    # Record the number of epochs since the best performance of the model
    epochs_since_last_best = {task_name: 0 for task_name, task in task_dict.items()}

    # Specify in the logs whether a given result is from fine tuning or multi-task training
    run_type_log_prefix = "FT " if is_fine_tuning else "MTL "

    # Get the required number of epochs for training
    epochs = PARAMS.num_fine_tuning_epochs if is_fine_tuning else PARAMS.num_epochs

    def is_patience_exceeded(task_name):
        return is_fine_tuning and epochs_since_last_best[task_name] >= PARAMS.early_stopping_patience

    # Start clock before training to measure how long it takes to find a validated best model
    train_time_start = time.clock()

    # Save initial model before training starts
    for task_name, task in task_dict.items():
        model_saver.save_model(file_name=task_name, model=task.model)

    for epoch in range(epochs):

        # Reset iterable for each task
        for task_name, task in task_dict.items():
            task.training_iterable = iter(task.train_data)

        # TRAIN
        for task_name in task_training_list:
            # Skip training this task if training patience already exceeded (during fine tuning only).
            # We do not skip on MTL training as there could be complex interactions between the training of multiple tasks.
            if is_patience_exceeded(task_name):
                print(f"{task_name} patience exceeded, ceasing training on this task")
                continue

            task = task_dict[task_name]

            X, y  = next(task.training_iterable)

            loss_fn = task.loss_fn()

            if PARAMS.cpu:
                logits = task.model(X.cpu())
                golds = y.cpu()
            else:
                logits = task.model(X.cuda())
                golds = y.cuda()

            if task.is_multilabel:
                loss = loss_fn(logits.view(-1, task.n_classes), golds)
            else:
                loss = loss_fn(logits.view(-1, task.n_classes), golds.view(-1))

            loss.backward()
            task.optimizer.step()
            task.model.zero_grad()

            logger.log_metric(f'{run_type_log_prefix} {task_name} - loss', x=task_steps[task_name], y=loss.item())

            # Only log overall loss when the tasks have a shared language model layer. During fine tuning, their models are no longer shared, making this metric useless.
            if not is_fine_tuning:
                logger.log_metric(f'{run_type_log_prefix} overall loss', x=step_num, y=loss.item())

            step_num += 1
            task_steps[task_name] += 1

        # VALIDATE
        for task_name, task in task_dict.items():
            if is_patience_exceeded(task_name):
                print(f"{task_name} patience exceeded, ceasing evaluation on this task")
                continue

            validation_results = task_eval_engines[task_name].run(task.valid_data).metrics
            logger.log_results(run_type_log_prefix + task_name, "valid", epoch, validation_results)

            # What metric will we compare all previous performance against
            comparison_metric = validation_results[PARAMS.best_metric]

            if comparison_metric > max(task_eval_metrics[task_name]):
                model_saver.save_model(file_name=task_name, model=task.model)
                epochs_since_last_best[task_name] = 0
            else:
                epochs_since_last_best[task_name] += 1
            task_eval_metrics[task_name].append(comparison_metric)

    train_time_end = time.clock()

    task_eval_metrics["time_elapsed"] = train_time_end - train_time_start

    # TEST
    task_test_metrics = {task_name: None for task_name, task in task_dict.items()}
    for task_name, task in task_dict.items():
        model_saver.load_model(file_name=task_name, model=task.model)

        test_engine = create_eval_engine(model=task.model, is_multilabel=task.is_multilabel, n_classes=task.n_classes, cpu=PARAMS.cpu)
        test_results = test_engine.run(task.test_data).metrics

        task_test_metrics[task_name] = test_results

        epoch = 1 if is_fine_tuning else 0
        logger.log_results(run_type_log_prefix + task_name, "test", epoch, test_results)

    return task_eval_metrics, task_test_metrics
