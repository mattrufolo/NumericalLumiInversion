# From  Hamish Graham

import json
import datetime

# load the configuration
def read_json(myfile, verbose=False):
    """
    Read a json file and convert it into python.

    Example
    --------
    >>> read_json('mytest.json')

    This will return the contents inside of a json file.
    """
    try: 
        with open(myfile, 'r') as file:
            my_dict = json.load(file)
        return my_dict
    except FileNotFoundError:
        if verbose: print('New file created.')
        my_dict = {}
        return my_dict
    except Exception as e: 
        if verbose: print(e.__class__)
        return None


def write_json(my_dict, myfile):
    """
    Convert a dictionary into a json file.
     
    Examples
    --------
    >>> write_json({'green': 'hello'}, ('mytest2.json'))

    This will add {'green': 'hello'} as a dictionary to the ('mytest2.json') file.
    """     
    with open(myfile, 'w') as file:  
        json.dump(my_dict, file, indent=2)


def append_json(my_dict, myfile):
    '''
    Append dictionaries to a json file.
    
    Examples
    --------
    >>> append_json({'blue': 'bonjour'}, ('mytest2.json'))

    This will append {'blue': 'bonjour'} to an existing json file: ('mytest2.json')
    '''
        
    try:
        with open(myfile, "r") as file:
            data = json.load(file)
            data.update(my_dict)
    except:
        data=my_dict
    write_json(data, myfile)

def get_last_stage(myfile, verbose=False):
    """
    get_last_stage is to read a json file and to return the number of the last dictionary

    Examples
    --------
    >>> get_last_stage('myfile', verbose=True)

    If the get_last_stage has two dictionaries: labeled '0' and '1' it will return the '1', the last stage.
    """     
    my_dict=read_json(myfile, verbose)
    try:
        return int(list(my_dict.keys())[-1])+1
    except IndexError:
        if verbose: print('IndexError, I consider 0 as first item')
        return 0
    except Exception as e: 
        if verbose: print(e.__class__)
        return 0


def tag_it(myfile, mycomment):
    """
    Create timestamps and add them to a json file.
    
    Examples
    --------
    >>> tag_it('myfile', 'hello')

    This will create a a human readable and a unix time stamp and append that to 'myfile', 
    including the comment 'hello'
    """     
    #stage = get_last_stage(myfile)
    #my_dict = {stage: {}}
    #my_dict[stage]['tag'] = mycomment
    my_now=datetime.datetime.now()
    #my_dict[stage]['unix_time'] = int(1e9*my_now.timestamp())        #in nanoseconds
    #my_dict[stage]['human_time'] = str(my_now)
    #append_json(my_dict, myfile)
    append_json({mycomment: int(1e9*my_now.timestamp())}, myfile)
