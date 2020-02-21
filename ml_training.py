metrics = []

best_model = None

eval_engine = get_eval_engine_for_model(cls_lm)

step_num = 0

for epoch in range(epochs):
    for X, y in train_data:
        logits = cls_lm(X.cuda())
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, cls_lm.n_classes), y.cuda().view(-1))
        loss.backward()
        neptune.log_metric(log_name="split_num " + str(split_num) + ' - loss', x=step_num, y=loss.item())
        optimizer.step()
        cls_lm.zero_grad()
        step_num += 1

    eval_results = eval_engine.run(valid_data)

    write_results(eval_results, split_num, epoch=epoch)

    print(eval_results.metrics["confusion_matrix"].numpy())

    if eval_results.metrics["accuracy"] > max([0] + [x["accuracy"] for x in metrics]):
        best_model = copy.deepcopy(cls_lm)

    metrics.append(eval_results.metrics)

test_engine = get_eval_engine_for_model(best_model)
test_results = test_engine.run(test_data)

write_results(test_results, split_num, epoch=None)

split_results.append(test_results.metrics)
