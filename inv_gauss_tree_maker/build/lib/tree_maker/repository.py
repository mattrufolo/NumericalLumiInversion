'''
This package helps to create download a update a list of git repository.
'''
# conda env export -n base > environment.yml
# conda env create --prefix /home/HPC/sterbini/test1 -f environment.yml  
# import git
import os
import subprocess
from pathlib import Path

# g = git.cmd.Git(git_dir)
# g.pull()

# https://stackoverflow.com/questions/14989858/
def get_hash(repo):
    ''' Given the repository path it gives the hash of the present git version.'''
    #os.chdir(str(Path(repo).expanduser()))
    return subprocess.check_output([f'git', '-C', f'{str(Path(repo).expanduser())}',
           'rev-parse', 'HEAD'])[0:-1].decode("utf-8") 


def create_base(my_list, verbose=False):
    '''Given a path a list of dictionary containint the *folder_name* and 
    the *repo* field (address of the repository) it *git clone* the 
    the repositories (or *git pull* if already cloned). 
    '''
    for ii in my_list:
        print(f'**** {ii["repo"]} to {ii["folder_name"]} ****')
        if Path(ii["folder_name"]).is_dir():
            if verbose:
                print("The folder exists.")
            try:
                os.system(f'git -C {ii["folder_name"]} pull')
            except Exception as e:
                print(e)
        else:
            if verbose:
                print("The folder does not exist exists.")
            try:
                os.system(f'git clone --depth=1 {ii["repo"]} {ii["folder_name"]}')
            except Exception as e:
                print(e)
        print("")


if __name__ == "__main__":
    #my_base = Path('~').expanduser() / "base"
    my_base = Path('/afs/cern.ch/eng')
    my_list = [
        {
            "repo": "http://github.com/lhcopt/lhcmask",
            "folder_name": str(my_base / "tracking-tools/modules"),
        },
        {
            "repo": "http://github.com/lhcopt/lhcerrors",
            "folder_name": str(my_base / "tracking-tools/errors"),
        },
        {
            "repo": "http://github.com/lhcopt/lhcmachines",
            "folder_name": str(my_base / "tracking-tools/machines"),
        },
        {
            "repo": "http://github.com/lhcopt/lhctoolkit",
            "folder_name": str(my_base / "tracking-tools/tools"),
        },
        {
            "repo": "http://github.com/lhcopt/beambeam",
            "folder_name": str(my_base / "tracking-tools/beambeam_macros"),
        },
        {
            "repo": "http://github.com/lhcopt/hllhc13",
            "folder_name": str(my_base / "lhc/optics/HLLHCV1.3"),
        },
        {
            "repo": "http://github.com/lhcopt/hllhc14",
            "folder_name": str(my_base / "lhc/optics/HLLHCV1.4"),
        },
        {
            "repo": "http://github.com/lhcopt/hllhc15",
            "folder_name": str(my_base / "lhc/optics/HLLHCV1.5"),
        },
        {
            "repo": "http://github.com/lhcopt/lhcrunIII",
            "folder_name": str(my_base / "lhc/optics/runIII"),
        },
        {
            "repo": "http://github.com/lhcopt/lhc2018",
            "folder_name": str(my_base / "lhc/optics/runII/2018"),
        },
        {
            "repo": "https://github.com/SixTrack/pysixtrack",
            "folder_name": str(my_base / "tracking-tools/python_installations/pysixtrack"),
        },
        {
            "repo": "https://github.com/SixTrack/sixtracktools",
            "folder_name": str(my_base
            / "tracking-tools/python_installations/sixtracktools"),
        },
        {
            "repo": "https://gitlab.cern.ch/abpcomputing/sandbox/tree_maker",
            "folder_name": str(my_base / "tracking-tools/python_installations/tree_maker"),
        },
    ]
    create_base(my_list, True)
    for ii in my_list:
        ii['hash'] = get_hash(ii['folder_name'])
