def fitness_classifiers(args, img):
    do_score_reverse = False
    if 'MODEL_REVERSE' in os.environ:
        print("-> predictions reversed")
        do_score_reverse = True

    active_model_keys = sorted(args.active_models.keys())

    # build a table indexed by target_size for all resized image lists
    target_size_table = {}
    for k in active_model_keys:
        model = args.active_models[k]
        target_size = model.get_target_size()
        target_size_table[target_size] = None

    for target_size in target_size_table:
        if target_size is None:
            imr = img
        else:
            imr = img.resize(target_size, resample=Resampling.BILINEAR)
        # target_size_table[target_size].append(tf.keras.utils.img_to_array(imr))
        target_size_table[target_size] = imr

    # convert all lists to np arrays
    # for target_size in target_size_table:
    #     target_size_table[target_size] = np.array(target_size_table[target_size])

    # make all predictions
    full_predictions = []
    fitness_partials = {}
    for k in active_model_keys:
        model = args.active_models[k]
        target_size = model.get_target_size()
        image_preprocessor = model.get_input_preprocessor()

        images = target_size_table[target_size]
        # images = np.copy(target_size_table[target_size])
        if image_preprocessor is not None:
            batch = image_preprocessor(images).unsqueeze(0)
        else:
            batch = images
        preds = model.predict(batch)
        # print("PREDS:", preds.shape, preds)
        if isinstance(preds, dict) and "scores" in preds:
            # print(preds['scores'].shape)
            if len(preds['scores'].shape) == 1:
                worthy = preds['scores']
            elif preds['scores'].shape[1] == 1:
                worthy = preds['scores']
            else:
                worthy = preds['scores'][:, args.imagenet_indexes]
        else:
            worthy = preds[:, args.imagenet_indexes]
            # print("Worthy {}: {}".format(k, np.array(worthy).shape))
            full_predictions.append(worthy)
            fitness_partials[k] = float(worthy)

            # convert predictions to np array
        full_predictions = torch.stack(full_predictions)
        if do_score_reverse:
            print("-> Applying predictions reversed")
            full_predictions = 1.0 - full_predictions

        # top_classes = np.argmax(full_predictions, axis=2).flatten()
        top_classes = torch.argmax(full_predictions, dim=2).flatten()
        # top_class = np.argmax(np.bincount(top_classes))
        top_class = torch.argmax(torch.bincount(top_classes))
        imagenet_index = args.imagenet_indexes[top_class]

        prediction_list = torch.sum(full_predictions, dim=2)

        # extract rewards and merged
        # rewards = np.sum(np.log(prediction_list + 1), axis=0)
        rewards = torch.sum(torch.log(prediction_list + 1), dim=0)
        # merged = np.dstack(prediction_list)[0]
        # merged = torch.dstack(prediction_list)[0]

        return rewards[0]