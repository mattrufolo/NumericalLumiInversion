import numpy as np
import pandas as pd


def search_boundary(data):
    events_ALICE = []
    events_LHCb = []
    boundary_dict = {
                "idx" : [],
                "events_LHCb" : [],
                "events_ALICE" : [],
                }
    #print("error")
    if len(data.index) == 1:
        bound_df = [data.iloc[0]]
    else:
        #print(data)
        events_ALICE = np.append(events_ALICE,data['events_ALICE'])
        events_LHCb = np.append(events_LHCb,data['events_LHCb'])

        for i in range(len(data['events_LHCb'])):
            #print(data['events_LHCb'][i])
            #print(boundary_dict["events_LHCb"])
            if data['events_LHCb'].iloc[i] in boundary_dict["events_LHCb"]:
                
                bound_index = np.where(boundary_dict["events_LHCb"] == np.array([data['events_LHCb'].iloc[i]]))[0][0]
                #print(boundary_dict["events_LHCb"])
                #print(bound_index)
                if data['events_ALICE'].iloc[i] > boundary_dict["events_ALICE"][bound_index]:
                    #print("aspie")
                    boundary_dict["idx" ][bound_index]  = i
                    boundary_dict["events_ALICE"][bound_index] = data['events_ALICE'].iloc[i]
            else:
                boundary_dict["idx" ].append(i)
                boundary_dict["events_LHCb"].append(data['events_LHCb'].iloc[i])
                boundary_dict["events_ALICE"].append(data['events_ALICE'].iloc[i])
        bound_df = [data.iloc[boundary_dict["idx" ][ii]] for ii in np.arange(len(boundary_dict["idx"]))]
        
        #print(bound_df)
        
    return bound_df


        