from keras.layers import Embedding, Input, dot, concatenate
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from IPython.display import SVG
from keras.utils import model_to_dot
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint




def model_architecture(n_users, n_movies, n_latent_factors):
    # Model Architecture

    # User Embeddings
    user_input = Input(shape=(1,), name='User_Input')
    user_embeddings = Embedding(input_dim = n_users, output_dim=n_latent_factors, input_length=1, 
                                name='User_Embedding') (user_input)
    user_vector = Flatten(name='User_Vector') (user_embeddings)


    # Movie Embeddings
    movie_input = Input(shape=(1,), name='Movie_Input')
    movie_embeddings = Embedding(input_dim = n_movies, output_dim=n_latent_factors, input_length=1, 
                                name='Movie_Embedding') (movie_input)
    movie_vector = Flatten(name='Movie_Vector') (movie_embeddings)


    # Dot Product
    merged_vectors = dot([user_vector, movie_vector], name='Dot_Product', axes=1)
    model = Model([user_input, movie_input], merged_vectors)

    return model


def model_visualization(model):
    SVG(model_to_dot( model,  show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))

