import numpy as np
import tensorflow as tf
from numpy.lib.recfunctions import structured_to_unstructured
from tensorflow import keras
import itertools


# Check whether TensorFloat-32 execution is currently enabled
tf.config.experimental.enable_tensor_float_32_execution(False)

class DataGeneratorTwoType(keras.utils.Sequence):
    """
    Custom Keras data generator for simulation data with a sliding window.
    To be passed to the fit_generator method of a Keras model.
    Follow template from https://github.com/afshinea/keras-data-generator 
    
    Parameters
    ----------
    simData : dict
        Dictionary containing simulation data.
    
    batch_size : int
        Number of samples per batch.
    
    shuffle : bool
        Whether to shuffle the data at the end of each epoch.
    
    inputKeys1 : list
        List of keys for input data 1.
    
    inputKeys2 : list
        List of keys for input data 2.
    
    outputKeys1 : list
        List of keys for output data 1.
        
    outputKeys2 : list  
        List of keys for output data 2.
        
    windowSigma : float
        Width of the sliding window
    """

    def __init__(
            self, 
            simData, 
            batch_size=32, 
            shuffle=True, 
            inputKeys1=["rho_H"], 
            inputKeys2=["rho_O"], 
            outputKeys=["c1_H"], 
            binKey="xbins",
            windowSigma=2.0):
        """
        Initializes DataGenerator with given parameters.
        """
        self.simData = simData
        self.inputKeys1 = inputKeys1
        self.inputKeys2 = inputKeys2
        self.outputKeys = outputKeys
        self.windowSigma = windowSigma
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        firstSimData = list(self.simData.values())[0]
        self.dz = 2 * firstSimData[binKey][0]
        self.simDataBins = len(firstSimData[binKey])
        self.windowBins = int(round(self.windowSigma / self.dz))
        
        self.inputData1 = {}
        self.inputData2 = {}
        self.outputData = {}
        self.validBins = {}
        
        self._prepare_data()
        
        self.input1Shape = (2 * self.windowBins + 1, len(self.inputKeys1))
        self.input2Shape = (2 * self.windowBins + 1, len(self.inputKeys2))
        self.outputShape = (len(self.outputKeys),)
        
        
        self.on_epoch_end()
        
        print(f"Initialized DataGenerator with {len(self.simData)} simulations,")
        print(f"yielding up to {len(self.indices)} samples in batches of {self.batch_size}")
    
    def _prepare_data(self):
        """
        Prepares the input and output data for each simulation.
        """
        for simId, data in self.simData.items():
            currentSimData = self.simData[simId]
            self.simDataBins = len(currentSimData["xbins"])
            valid = np.full(self.simDataBins, True)
            
            for k in self.outputKeys :
                if len(data[k]) != self.simDataBins:
                    print(f"Invalid data length for key {k} in simulation {simId}")
                    pass
                valid = np.logical_and(valid, ~np.isnan(data[k]))
            self.validBins[simId] = np.flatnonzero(valid)

            self.inputData1[simId] = structured_to_unstructured(
                np.pad(data[self.inputKeys1], self.windowBins, mode="wrap"))
            self.inputData2[simId] = structured_to_unstructured(
                np.pad(data[self.inputKeys2], self.windowBins, mode="wrap"))
            self.outputData[simId] = structured_to_unstructured(
                np.pad(data[self.outputKeys], self.windowBins, mode="wrap"))
          
    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples.
        """
        X1 = np.empty((self.batch_size, *self.input1Shape))
        X2 = np.empty((self.batch_size, *self.input2Shape))
        y = np.empty((self.batch_size, *self.outputShape))

        
        for b, (simId, i) in enumerate(indexes):
            i += self.windowBins
            X1[b] = self.inputData1[simId][i - self.windowBins:i + self.windowBins + 1]
            X2[b] = self.inputData2[simId][i - self.windowBins:i + self.windowBins + 1]
            y[b] = self.outputData[simId][i]
            
        return (X1, X2), y

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indices after each epoch.
        """
        self.indices = [
            (simId, i) 
            for simId in self.simData.keys() 
            for i in self.validBins[simId]
        ]
        if self.shuffle:
            np.random.default_rng().shuffle(self.indices)

    def __getitem__(self, index):
        """
        Generates one batch of data.
        """
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        data = self.__data_generation(indexes)

        return data



class DataGeneratorSingleType(keras.utils.Sequence):
    """
    Custom Keras data generator for simulation data with a sliding window.
    To be passed to the fit_generator method of a Keras model.
    Follow template from https://github.com/afshinea/keras-data-generator 
    
    Parameters
    ----------
    simData : dict
        Dictionary containing simulation data.
    
    batch_size : int
        Number of samples per batch.
    
    shuffle : bool
        Whether to shuffle the data at the end of each epoch.
    
    inputKeys : list
        List of keys for input data 1.
    
    outputKeys : list
        List of keys for output data 1.

    windowSigma : float
        Width of the sliding window
    """

    def __init__(
            self, 
            simData, 
            batch_size=32, 
            shuffle=True, 
            inputKeys=["rho"], 
            outputKeys=["c1"], 
            binKey="xbins",
            windowSigma=2.0):
        """
        Initializes DataGenerator with given parameters.
        """
        self.simData = simData
        self.inputKeys = inputKeys
        self.outputKeys = outputKeys
        self.windowSigma = windowSigma
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        firstSimData = list(self.simData.values())[0]
        self.dz = 2 * firstSimData[binKey][0]
        self.simDataBins = len(firstSimData[binKey])
        self.windowBins = int(round(self.windowSigma / self.dz))
        
        self.inputData = {}
        self.outputData = {}
        self.validBins = {}
        
        self._prepare_data()
        
        self.inputShape = (2 * self.windowBins + 1, len(self.inputKeys))
        self.outputShape = (len(self.outputKeys),)
        
        self.on_epoch_end()
        
        print(f"Initialized DataGenerator with {len(self.simData)} simulations,")
        print(f"yielding up to {len(self.indices)} samples in batches of {self.batch_size}")
    
    def _prepare_data(self):
        """
        Prepares the input and output data for each simulation.
        """
        for simId, data in self.simData.items():
            valid = np.full(self.simDataBins, True)
            for k in self.outputKeys:
                valid = np.logical_and(valid, ~np.isnan(data[k]))
            self.validBins[simId] = np.flatnonzero(valid)

            self.inputData[simId] = structured_to_unstructured(
                np.pad(data[self.inputKeys], self.windowBins, mode="wrap"))
            self.outputData[simId] = structured_to_unstructured(
                np.pad(data[self.outputKeys], self.windowBins, mode="wrap"))
            
    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples.
        """
        X = np.empty((self.batch_size, *self.inputShape))
        y = np.empty((self.batch_size, *self.outputShape))
        
        for b, (simId, i) in enumerate(indexes):
            i += self.windowBins
            X[b] = self.inputData[simId][i - self.windowBins:i + self.windowBins + 1]
            y[b] = self.outputData[simId][i]
            
        return X, y

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indices after each epoch.
        """
        self.indices = [
            (simId, i) 
            for simId in self.simData.keys() 
            for i in self.validBins[simId]
        ]
        if self.shuffle:
            np.random.default_rng().shuffle(self.indices)

    def __getitem__(self, index):
        """
        Generates one batch of data.
        """
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        data = self.__data_generation(indexes)

        return data
    

def get_dataset_c1(trainingGenerator):
    def gen():
        for i in range(len(trainingGenerator)):
            yield trainingGenerator[i]

    return tf.data.Dataset.from_generator(gen, output_signature=(
        {
            "rho": tf.TensorSpec(shape=(trainingGenerator.batch_size, trainingGenerator.inputShape[0]), dtype=tf.float32),
        },
        {
            "c1": tf.TensorSpec(shape=(trainingGenerator.batch_size, 1), dtype=tf.float32),
        }
    )).prefetch(tf.data.AUTOTUNE)

def get_dataset_c1_O_twotype(trainingGenerator):
    def gen():
        for i in range(len(trainingGenerator)):
            yield trainingGenerator[i]

    return tf.data.Dataset.from_generator(gen, output_signature=(
        {
            "rho_O": tf.TensorSpec(shape=(trainingGenerator.batch_size, trainingGenerator.input1Shape[0]), dtype=tf.float32),
            "rho_H": tf.TensorSpec(shape=(trainingGenerator.batch_size, trainingGenerator.input2Shape[0]), dtype=tf.float32),
        },
        {
            "c1_O": tf.TensorSpec(shape=(trainingGenerator.batch_size, 1), dtype=tf.float32),
        }
    )).prefetch(tf.data.AUTOTUNE)



def get_dataset_c1_H_twotype(trainingGenerator):
    def gen():
        for i in range(len(trainingGenerator)):
            yield trainingGenerator[i]

    return tf.data.Dataset.from_generator(gen, output_signature=(
        {
            "rho_H": tf.TensorSpec(shape=(trainingGenerator.batch_size, trainingGenerator.input1Shape[0]), dtype=tf.float32),
            "rho_O": tf.TensorSpec(shape=(trainingGenerator.batch_size, trainingGenerator.input2Shape[0]), dtype=tf.float32),
        },
        {
            "c1_H": tf.TensorSpec(shape=(trainingGenerator.batch_size, 1), dtype=tf.float32),
        }
    )).prefetch(tf.data.AUTOTUNE)


def get_dataset_c1b(pc_data, inputShape):
    def calc_c1b(rho, mu, T):
        return np.log(rho) - mu / T

    training_inputs_c1b = {
        "rho": np.array([np.full(inputShape, pc["rhob"]) for pc in pc_data]),
    }

    training_output_c1b = {
        "c1": np.array([calc_c1b(pc["rhob"], pc["mu"], pc["T"]) for pc in pc_data if np.isfinite(calc_c1b(pc["rhob"], pc["mu"], pc["T"]))])
    }

    return tf.data.Dataset.from_tensor_slices((training_inputs_c1b, training_output_c1b)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).repeat(64).batch(16).prefetch(tf.data.AUTOTUNE)


def get_dataset_c2x(pc_data, windowSigma, inputShape):
    def construct_c2x(c2x, xs):
        c2x = c2x[xs<windowSigma]
        result = np.concatenate((c2x[:0:-1], c2x))
        if result.shape != inputShape:
            raise ValueError(f"The pair-correlation matching is not commensurable with the model. Is the discretization the same?")
        return result

    training_inputs_pc = {
        "rho": np.array([np.full(inputShape, pc["rhob"]) for pc in pc_data]),
    }

    training_output_pc = {
        "c2": np.array([construct_c2x(pc["c2x"], pc["xs"]) for pc in pc_data])
    }

    return tf.data.Dataset.from_tensor_slices((training_inputs_pc, training_output_pc)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).repeat().batch(16).prefetch(tf.data.AUTOTUNE)


def get_dataset_c2_O_twotype(pc_data, windowSigma, input1Shape, input2Shape):
    def construct_c2x(c2x, xs):
        c2x = c2x[xs<windowSigma]
        result = np.concatenate((c2x[:0:-1], c2x))
       
        if result.shape != input1Shape:
            raise ValueError(f"The pair-correlation matching is not commensurable with the model. Is the discretization the same?")
        return result
    
    training_inputs_pc = {
        "rho_O": np.array([np.full(input1Shape, pc_data[pc]["rho_O"][0]) for pc in pc_data]),
        "rho_H": np.array([np.full(input2Shape, pc_data[pc]["rho_H"][0]) for pc in pc_data]),
    }
  
    training_output_pc = {
        "c2_OO": np.array([construct_c2x(pc_data[pc]["c2_OO"], pc_data[pc]["xs"]) for pc in pc_data]),
        "c2_OH": np.array([construct_c2x(pc_data[pc]["c2_OH"], pc_data[pc]["xs"]) for pc in pc_data]),
    }

    return tf.data.Dataset.from_tensor_slices((training_inputs_pc, training_output_pc)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).repeat().batch(16).prefetch(tf.data.AUTOTUNE)

def get_dataset_c2_H_twotype(pc_data, windowSigma, input1Shape, input2Shape):
    def construct_c2x(c2x, xs):
        c2x = c2x[xs<windowSigma]
        result = np.concatenate((c2x[:0:-1], c2x))
       
        if result.shape != input1Shape:
            raise ValueError(f"The pair-correlation matching is not commensurable with the model. Is the discretization the same?")
        return result
    
    training_inputs_pc = {
        "rho_H": np.array([np.full(input1Shape, pc_data[pc]["rho_H"][0]) for pc in pc_data]),
        "rho_O": np.array([np.full(input2Shape, pc_data[pc]["rho_O"][0]) for pc in pc_data]),
    }
  
    training_output_pc = {
        "c2_HH": np.array([construct_c2x(pc_data[pc]["c2_HH"], pc_data[pc]["xs"]) for pc in pc_data]),
        "c2_OH": np.array([construct_c2x(pc_data[pc]["c2_OH"], pc_data[pc]["xs"]) for pc in pc_data]),
    }

    return tf.data.Dataset.from_tensor_slices((training_inputs_pc, training_output_pc)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).repeat().batch(16).prefetch(tf.data.AUTOTUNE)


'''
Data generator which yields batches of input windows and output values from whole profiles (useful for local learning of neural functionals).
'''

class DataGenerator_inhom(keras.utils.PyDataset):
    def __init__(self, simData, batch_size=32, steps_per_execution=1, shuffle=True, inputKeys=["rho"], paramsKeys=[], outputKeys=["c1"], windowSigma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.simData = simData
        self.inputKeys = inputKeys
        self.paramsKeys = paramsKeys
        self.outputKeys = outputKeys
        self.windowSigma = windowSigma
        firstSimData = list(self.simData.values())[0]
        self.dz = 2 * firstSimData["profiles"]["z"][0]
        self.simDataBins = len(firstSimData["profiles"]["z"])
        self.windowBins = int(round(self.windowSigma/self.dz))
        self.validBins = {}
        self.inputDataPadded = {}
        for simId in self.simData.keys():
            valid = np.full(self.simDataBins, True)
            for k in self.outputKeys:
                valid = np.logical_and(valid, ~np.isnan(self.simData[simId]["profiles"][k]))
            self.validBins[simId] = np.flatnonzero(valid)
            self.inputDataPadded[simId] = np.pad(self.simData[simId]["profiles"][self.inputKeys], self.windowBins, mode="wrap")
        self.batch_size = batch_size
        self.steps_per_execution = steps_per_execution
        self.inputShape = (2*self.windowBins+1,)
        self.outputShape = (len(self.outputKeys),)
        self.shuffle = shuffle
        self.on_epoch_end()
        print(f"Initialized DataGenerator from {len(self.simData)} simulations which will yield up to {len(self.indices)} input/output samples in batches of {self.batch_size}")

    def __len__(self):
        return int(np.floor(len(self.indices) / (self.batch_size * self.steps_per_execution))) * self.steps_per_execution

    def __getitem__(self, index):
        ids = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        profiles = {key: np.empty((self.batch_size, *self.inputShape)) for key in self.inputKeys}
        params = {key: np.empty((self.batch_size, 1)) for key in self.paramsKeys}
        output = {key: np.empty((self.batch_size, *self.outputShape)) for key in self.outputKeys}
        for b, (simId, i) in enumerate(ids):
            for key in self.inputKeys:
                profiles[key][b] = self.inputDataPadded[simId][key][i:i+2*self.windowBins+1]
            for key in self.paramsKeys:
                params[key][b] = self.simData[simId]["params"][key]
            for key in self.outputKeys:
                output[key][b] = self.simData[simId]["profiles"][key][i]
        return (profiles | params), output

    def on_epoch_end(self):
        self.indices = []
        for simId in self.simData.keys():
            self.indices.extend(list(itertools.product([simId], list(self.validBins[simId]))))
        if self.shuffle == True:
            np.random.default_rng().shuffle(self.indices)

    def pregenerate(self):
        print("Pregenerating data from DataGenerator")
        batch_size_backup = self.batch_size
        self.batch_size *= len(self)
        data = self[0]
        self.batch_size = batch_size_backup
        return data
    
    
class DataGenerator_inhom_twotype(keras.utils.PyDataset):
    def __init__(self,
                 simData,
                 batch_size=32,
                 steps_per_execution=1,
                 shuffle=True,
                 inputKeys1=["rho_O"],
                 inputKeys2=["rho_H"],
                 paramsKeys=[],
                 outputKeys=["c1_O"],
                 binKey="xbins",
                 windowSigma=2.0,
                 **kwargs):
        '''
        Initializes DataGenerator with given parameters.
        '''
        
        super().__init__(**kwargs)
        self.simData = simData
        self.inputKeys1 = inputKeys1
        self.inputKeys2 = inputKeys2
        self.paramsKeys = paramsKeys
        self.outputKeys = outputKeys
        self.windowSigma = windowSigma
        
        firstSimData = list(self.simData.values())[0]
        self.dz = 2 * firstSimData[binKey][0]
        self.simDataBins = len(firstSimData[binKey])
        self.windowBins = int(round(self.windowSigma/self.dz))
        
        self.validBins = {}
        self.inputData1Padded = {}
        self.inputData2Padded = {}
        for simId in self.simData.keys():
            
            currentSimData = self.simData[simId]
            self.simDataBins = len(currentSimData["xbins"])
            valid = np.full(self.simDataBins, True)
                    
            for k in self.outputKeys:
                valid = np.logical_and(valid, ~np.isnan(self.simData[simId][k]))
            self.validBins[simId] = np.flatnonzero(valid)
            self.inputData1Padded[simId] = np.pad(self.simData[simId][self.inputKeys1], self.windowBins, mode="wrap")
            self.inputData2Padded[simId] = np.pad(self.simData[simId][self.inputKeys2], self.windowBins, mode="wrap")
        self.batch_size = batch_size
        
        self.steps_per_execution = steps_per_execution
        
        self.input1Shape = (2*self.windowBins+1,)
        self.input2Shape = (2*self.windowBins+1,)
        self.outputShape = (len(self.outputKeys),)
        
        self.shuffle = shuffle
        self.on_epoch_end()
        
        print(f"Initialized DataGenerator from {len(self.simData)} simulations which will yield up to {len(self.indices)} input/output samples in batches of {self.batch_size}")

    def __len__(self):
        return int(np.floor(len(self.indices) / (self.batch_size * self.steps_per_execution))) * self.steps_per_execution

    def __getitem__(self, index):
        """
        Generates one batch of data.
        """
        ids = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        inputkeys = self.inputKeys1 + self.inputKeys2
        
        profiles = {key: np.empty((self.batch_size, *self.input1Shape)) for key in inputkeys}
        #profiles2 = {key: np.empty((self.batch_size, *self.input2Shape)) for key in self.inputKeys2}
        params = {key: np.empty((self.batch_size, 1)) for key in self.paramsKeys}
        output = {key: np.empty((self.batch_size, *self.outputShape)) for key in self.outputKeys}
        for b, (simId, i) in enumerate(ids):
            profiles[inputkeys[0]][b] = self.inputData1Padded[simId][i:i+2*self.windowBins+1]
            profiles[inputkeys[1]][b] = self.inputData2Padded[simId][i:i+2*self.windowBins+1]
            
            for key in self.paramsKeys:
                params[key][b] = self.simData[simId]["params"][key]
            for key in self.outputKeys:
                output[key][b] = self.simData[simId][key][i]
        return ((profiles | params)),  output

    def on_epoch_end(self):
        self.indices = []
        for simId in self.simData.keys():
            self.indices.extend(list(itertools.product([simId], list(self.validBins[simId]))))
        if self.shuffle == True:
            np.random.default_rng().shuffle(self.indices)

    def pregenerate(self):
        print("Pregenerating data from DataGenerator")
        batch_size_backup = self.batch_size
        self.batch_size *= len(self)
        data = self[0]
        self.batch_size = batch_size_backup
        return data
    