from anytree import AnyNode
from anytree.exporter import DictExporter
from anytree.importer import DictImporter
from tree_maker import NodeJob
import yaml # pip install pyyaml 
import ruamel.yaml
import json
import orjson

ryaml = ruamel.yaml.YAML()

def tree_from_yaml(filename='tree.yaml'): 
    '''Import the tree structure from the yaml formatted *filename*'''
    with open(filename, "r") as file:
        return DictImporter(nodecls=NodeJob).import_(yaml.load(file, Loader=yaml.FullLoader))
        
def tree_from_json(filename='tree.json'): 
    '''Import the tree structure from the json formatted *filename*'''
    with open(filename, "r") as file:
        return DictImporter(nodecls=NodeJob).import_(orjson.loads(file.read()))
        
def from_yaml(filename):
    '''Load the *filename* yaml file'''
    try:
        with open(filename, 'r') as file:
            return ryaml.load(file)
    except Exception as e:
        print(e)
        return {}

def from_json(filename, verbose=False):
    '''Load the *filename* json file'''
    try:
        with open(filename, 'r') as file:
            return orjson.loads(file.read())
    except Exception as e:
        if verbose: print(e)
        return {}

def config_to_yaml():
    '''Convert the config.py (containg dictionaries) to an
    equivalent config.yaml'''
    import config as my_dict
    with open('config.yaml', 'w') as fid:
        yaml.dump(my_dict.configuration, fid)
