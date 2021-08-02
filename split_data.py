def split_data(pandas_df,train_size):

    if ((train_size > 0) & (train_size <=1)):

        pass

    elif train_size > 1:

        train_size = train_size/pandas_df.shape[0]

    train_set = pandas_df.sample(frac = train_size,random_state = 42)

    test_set = pandas_df.drop(train_set.index).reset_index(drop = True)

    train_set = train_set.reset_index(drop = True)

    return train_set,test_set

