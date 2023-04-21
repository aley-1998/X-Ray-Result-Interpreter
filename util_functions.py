def unzip_zip():

    to_extract = input('Type the path of the file to unzip:')
    path = to_extract.split('/')
    path.pop(-1)
    extract_in = '/'.join(map(str,path))
    with ZipFile(to_extract, 'r') as zObject:
        zObject.extractall(extract_in)

def data_split(dataframes, train_split=0.8):
    ''' 
    (list, float) -> pandas.core.frame.DataFrame
    
    dataframes: list of dataframes
    train_split: training set ratio (default = 0.8)
    
    
    Takes a list of df's (dataframes) and a float (train_split) as input. Returns all df's under one df, 
    indicating whether a row is marked for training, testing or validation (testing and validation are of equal size).
    '''
    
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    val_set = pd.DataFrame()
    
    for df in dataframes:
        df = shuffle(df)
        
        # train data
        df.reset_index(drop=True, inplace=True)
        train = pd.DataFrame(df[:round(train_split*len(df))-1])
        train_set = pd.concat([train_set, train], ignore_index=True)
        
        # test data
        df.reset_index(drop=True, inplace=True)
        test = pd.DataFrame(df[:round(len(df)/2)]) 
        test_set = pd.concat([test_set, test], ignore_index=True)
        
        # val data
        df.reset_index(drop=True, inplace=True)
        val = pd.DataFrame(df[round(train_split*len(df)):round(len(df)/2)])
        train_set = pd.concat([train_set, train], ignore_index=True)
        
    train_set = shuffle(train_set.assign(SPLIT = lambda x: ('train')))
    test_set = shuffle(test_set.assign(SPLIT = lambda x: ('test')))
    val_set = shuffle(val_set.assign(SPLIT = lambda x: ('valid')))
    
    df = pd.concat([train_set, test_set, val_set], ignore_index=True) # combine all 3 dfs
    df.reset_index(drop=True, inplace=True)
    
    return df
 
def file_paths():
    
    # Creating 2 columns containing filepaths for image and mask files
    
    normal_df = pd.read_excel(r'data\Normal.metadata.xlsx').assign(PATH_IMAGES = lambda x: ('/data/images_unf/Normal/' + x['FILE NAME']))
    normal_df = normal_df.assign(PATH_MASKS = lambda x: ('/data/mask_data/Normal/' + x['FILE NAME']))
    normal_df = normal_df[['FILE NAME', 'PATH_IMAGES', 'PATH_MASKS']]
    
    covid_df = pd.read_excel(r'data\COVID.metadata.xlsx').assign(PATH_IMAGES = lambda x: ('/data/images_unf/COVID/' + x['FILE NAME']))
    covid_df = covid_df.assign(PATH_MASKS = lambda x: ('/data/mask_data/COVID/' + x['FILE NAME']))
    covid_df = covid_df[['FILE NAME', 'PATH_IMAGES', 'PATH_MASKS']]
    
    opacity_df = pd.read_excel(r'data\Lung_Opacity.metadata.xlsx').assign(PATH_IMAGES = lambda x: ('/data/images_unf/Lung_Opacity/' + x['FILE NAME']))
    opacity_df = opacity_df.assign(PATH_MASKS = lambda x: ('/data/mask_data/Lung_Opacity/' + x['FILE NAME']))
    opacity_df = opacity_df[['FILE NAME', 'PATH_IMAGES', 'PATH_MASKS']]
    
    pneumonia_df = pd.read_excel(r'data\Viral Pneumonia.metadata.xlsx').assign(PATH_IMAGES = lambda x: ('/data/images_unf/Viral Pneumonia/' + x['FILE NAME']))
    pneumonia_df = pneumonia_df.assign(PATH_MASKS = lambda x: ('/data/mask_data/Viral Pneumonia/' + x['FILE NAME']))
    pneumonia_df = pneumonia_df[['FILE NAME', 'PATH_IMAGES', 'PATH_MASKS']]
    
    return [normal_df, covid_df, opacity_df, pneumonia_df]

def evaluate_model(model, n, valid, test):
    
    pd.DataFrame(model.history.history).plot(figsize=(15,5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title(f'Model {n} Learning Curves')
    plt.show()
    print()
    print()
    print()
    print('Test set score')
    print()
    model.evaluate(test)
    print()
    print()
    print()
    print('Validation set score')
    print()
    model.evaluate(valid)
    
def save_model(model, n):
    
    model.save(f"model{n}.h5")
    history = pd.DataFrame(model.history.history)
    hist_csv_file = f"model{n}_history.csv"
    with open(hist_csv_file, mode='w') as f:
        history.to_csv(f)

def train_model(model, n, epochs, patience=5):

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    csv_logger = CSVLogger(f'model{n}_training.log')
    model.fit(train, validation_data=valid, verbose=1, epochs=epochs, shuffle=True, callbacks=[early_stopping_cb, csv_logger])

def network_builder(shape=(256, 256, 3),
                    layers=2,
                    neurons=200,
                    kernel_size=3,
                    layer_activ="relu",
                    reg=0.01, 
                    lr=0.01,
                    init_layer="he_normal",
                    pool=2,
                    #num_classes=2,
                    output_activ=None,
                    loss = "binary_crossentropy",
                    metrics = ['binary_accuracy']):

    # Network
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=shape)) # input
    model.add(keras.layers.Rescaling(1./255, offset=0.0))
    for i in range(layers): # layers
        model.add(keras.layers.Conv2D(neurons, kernel_size=kernel_size, activation=layer_activ, kernel_regularizer=keras.regularizers.l1(reg), kernel_initializer=init_layer))
        model.add(keras.layers.MaxPooling2D(pool_size=pool))
    model.add(keras.layers.Flatten()) # feature map
    model.add(keras.layers.Dense(1, activation=output_activ)) # output

    # Compiler
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss=loss,
              metrics=metrics)
    return model

def model_builder(hp):

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(256, 256, 3))) # input
    model.add(keras.layers.Rescaling(1./255, offset=0.0))
    
    hp_filters = hp.Int('filters', min_value=32, max_value=512, step=32)
    #hp_kernel_size = hp.Choice('kernel_size', values=[2, 3])
    #hp_pool_size = hp.Choice('pool_size', values=[2, 3, 4])
    model.add(keras.layers.Conv2D(filters=hp_filters, 
                                  kernel_size=3, 
                                  activation='relu', 
                                  kernel_regularizer=keras.regularizers.l1(0.01), 
                                  kernel_initializer="he_normal")) # hidden layer
    model.add(keras.layers.MaxPooling2D(pool_size=3))
    model.add(keras.layers.Flatten()) # feature map
    model.add(keras.layers.Dense(1)) # output

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=['binary_accuracy'])
    return model
