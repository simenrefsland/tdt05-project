import fine_tune
import pre_train

import argparse
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--savedmodel', action='store_true', help='Use saved model')
    arg('--modelpath', type=str, default='models/encoder.h5', help='Path to saved model')
    arg('--visualize', action='store_true', help='Visualize augmented data')
    arg('--fraction', type=float, default=0.01, help='Fraction of training data to use')
    
    args = parser.parse_args()
    saved_model = args.savedmodel
    model_path = args.modelpath
    visualize = args.visualize
    fraction = args.fraction
    
    if not saved_model:
        pre_train.pre_train_model(visualize)
        
    pre_trained_model = tf.keras.models.load_model(model_path)
    fine_tune.fine_tune(pre_trained_model, fraction)
    
if __name__ == "__main__":
    main()
    
    