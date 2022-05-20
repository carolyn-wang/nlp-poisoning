def label_to_float(k, v, label_col='labels'):
	if k == label_col:
		return v.float()
	return v
