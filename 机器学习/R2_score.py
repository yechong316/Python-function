def R2_score(y_true, y_pred):
    assert y_pred.shape == y_true.shape, '输入的{}与{}不相等'.format(y_pred.shape, y_true.shape)
	return np.sum(np.square(y_pred - y_true)) / np.sum(np.square(y_pred - np.mean(y_true)))