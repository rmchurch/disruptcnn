import torch

def validation_loss(T_min_warn=None,T_max_warn=None):
  """
  T_min_warn: Minimum time before disruption before considering as "missed"
  Twarn: The time before disruption we wish to begin labelling "disruptive"
  """
  if T_min_warn is None:
    T_min_warn = 30 #ms minimum warning time
  if T_max_warn is None:
    #TODO: This should be from the EceiDataset class "Twarn"
    T_max_warn = 300 #ms maximum warning time

