#import tensorflow as tf

#import os; 
#print(os.environ['PATH'].split(';'), sep='\n')
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



#import tensorflow

#tensorflow.test.is_built_with_gpu_support() # Test install of pip package, should output True
#tensorflow.test.is_gpu_available() # Should output True
#tensorflow.test.gpu_device_name() # Should output something like /device:GPU:0


#if tf.test.gpu_device_name():
#      print(f'Default GPU Device:{tf.test.gpu_device_name()}')
#else:
#      print("Please check your installation")
