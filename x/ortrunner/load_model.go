package ortrunner

type modelResource interface{ Close() }

func loadConfiguredModel[C, M, T modelResource](
	newConfig func() (C, error),
	configure func(C) error,
	newModel func(C) (M, error),
	newTokenizer func(M) (T, error),
) (config C, model M, tokenizer T, err error) {
	config, err = newConfig()
	if err != nil {
		return
	}
	keep := false
	defer func() {
		if !keep {
			config.Close()
		}
	}()
	if err = configure(config); err != nil {
		return
	}
	model, err = newModel(config)
	if err != nil {
		return
	}
	defer func() {
		if !keep {
			model.Close()
		}
	}()
	tokenizer, err = newTokenizer(model)
	if err != nil {
		return
	}
	keep = true
	return
}
