from fastapi import FastAPI
from classicalgsg import *
import sys
import os.path as osp
import numpy as np
from joblib import load

import torch
from classicalgsg.molreps_models.gsg import GSG
from classicalgsg.classicalgsg import OBFFGSG
from classicalgsg.molreps_models.utils import scop_to_boolean

PRETRAINED_MODEL_PATH = 'pretrained_models'
PRETRAINED_MODEL = 'model_4_zfs_MMFF.pt'
SCALER = 'std_scaler_MMFF.sav'
FORCEFIELD = 'MMFF94'

app = FastAPI()

@app.get("/")
async def root():
    return{"Enter SMILES:"}

@app.get("/api/{logpSMILES}")
async def read_item(logpSMILES: str):
    calculatedLogP = calculateLogP(logpSMILES)
    return {"logP: " + str(calculatedLogP)}

def calculateLogP(logpSMILES):
  max_wavelet_scale = 4
  scattering_operators = '(z,f,s)'

  gsg = GSG(max_wavelet_scale, scop_to_boolean(scattering_operators))
  uffgsg = OBFFGSG(gsg, structure='2D', AC_type='ACall')

  x = uffgsg.features(logpSMILES, FORCEFIELD)

  if x is not None:

        x = x.reshape((-1, x.shape[0]))

        scaler_file_path = osp.join(osp.dirname(sys.modules[__name__].__file__),
                                    PRETRAINED_MODEL_PATH,
                                    SCALER)

        model_file_path = osp.join(osp.dirname(sys.modules[__name__].__file__),
                                   PRETRAINED_MODEL_PATH,
                                   PRETRAINED_MODEL)

        scaler = load(scaler_file_path)
        model = torch.load(model_file_path)
        x = scaler.transform(x)

        predicted_logP = np.squeeze(model.predict(x.astype(np.float32)))

        #print(f'Predicted logP value is: {predicted_logP:.2f}')
        return predicted_logP

  else:
        print(f'Can not convert {smiles} to a openbabel molecule ')