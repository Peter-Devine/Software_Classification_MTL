import random
from evaluation_engines import create_eval_engine
from model_saver import ModelSaver

def train_on_tasks(task_dict, PARAMS, logger, is_fine_tuning):
    model_saver = ModelSaver(".")

    metrics = []

    task_eval_metrics = {task_name: [0] for task_name, task in task_dict.items()}
    task_steps = {task_name: 0 for task_name, task in task_dict.items()}
    task_eval_engines = {task_name: create_eval_engine(model=task.model, is_multilabel=task.is_multilabel) for task_name, task in task_dict.items()}

    task_training_list = []
    for task_name, task in task_dict.items():
        task_training_list.extend([task_name]*task.train_length)

    random.seed(PARAMS.random_state)

    if not is_fine_tuning:
        # Shuffle task list during multi-task training so that tasks are trained roughly evenly throughout
        random.shuffle(task_training_list)

    step_num = 0
    run_type_log_prefix = "Fine-tuning " if is_fine_tuning else "Multi-task training "

    epochs = PARAMS.num_fine_tuning_epochs if is_fine_tuning else PARAMS.num_epochs

    for epoch in range(epochs):
        # TRAIN
        for task_name in task_training_list:
            task = task_dict[task_name]

            loss_fn = task.loss_fn

            X, y  = next(task.train_data)

            logits = task.model(X.cuda())
            loss_fn = task.loss_fn()
            loss = loss_fn(logits.view(-1, task.n_classes), y.cuda().view(-1))
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
            validation_results = task_eval_engines[task_name].run(task.valid_data).metrics
            logger.log_results(run_type_log_prefix + task_name, "valid", epoch, validation_results)

            # What metric will we compare all previous performance against
            comparison_metric = validation_results["accuracy"]

            if comparison_metric > max(task_eval_metrics[task_name]):
                model_saver.save_model(file_name=task_name, task.model)

            task_eval_metrics[task_name].append(comparison_metric)

    # TEST
    task_test_metrics = {task_name: None for task_name, task in task_dict.items()}
    for task_name, task in task_dict.items():
        task.model = model_saver.load_model(file_name=task_name)

        test_engine = create_eval_engine(model=best_model, is_multilabel=task.is_multilabel)
        test_results = test_engine.run(task.test_data).metrics

        task_test_metrics[task_name] = test_results

        epoch = 1 if is_fine_tuning else 0
        logger.log_results(run_type_log_prefix + task_name, "test", epoch, test_results)

    return task_eval_metrics, task_test_metrics
