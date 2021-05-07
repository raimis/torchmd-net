import os
import stat
import numpy as np

def mopac_write_input(filename, atomtypes, coords, optimize, method='PM7', forces=False):

    assert not (optimize and forces), '"optimize" and "forces" cannot be enabled together'

    fout = open(filename, "w")
    # If there is nothing to optimize, the forces are not computed,
    # but setting 1SCF prevents from doing actual optimization.
    optimize = True if forces else optimize
    forces = '1SCF GRAD' if forces else ''
    fout.write(f"{method} {forces}\n\n\n")
    o = 1 if optimize else 0
    format = f"%-2s %18.15g {o} %18.15g {o} %18.15g {o}\n"
    for r, atom in zip(coords, atomtypes):
        fout.write(format % (atom, r[0], r[1], r[2]))
    fout.write('\n')
    fout.close()

def parse_configuration(lines):
    atoms_coords = []
    elements = []
    for l in lines[4:]:
        try:
            tok = l.split()
            elements.append(tok[0])
            atoms_coords.append((float(tok[1]),float(tok[3]),float(tok[5])))
        except: #finished reading
            break
    return elements, atoms_coords

def parse_energy(line):
    return line

def parse_forces(lines):

    forces = []
    for line in lines[3:]:
        try:
            forces.append(float(line.split()[6]))
        except:
            break
    forces = -np.array(forces).reshape((-1, 3))

    return forces

def mopac_read_arc(filename):
    '''
    Unitis:
      conf: Ang
      energy: eV
      forces: kcal/mol/Ang
    '''
    fin = open(filename,'r')
    txt = str(fin.read())
    start = txt.find('FINAL GEOMETRY OBTAINED')
    elem, conf = parse_configuration(txt[start:].split('\n'))
    start = txt.find('TOTAL ENERGY')
    energy = float(txt[start:].split('\n',1)[0].split()[3])
    start = txt.find('FINAL  POINT  AND  DERIVATIVES')
    forces = None
    if start != -1:
        forces = parse_forces(txt[start:].split('\n'))
    fin.close()
    return elem, conf, energy, forces

def mopac_write_rundir(dirname, atomtypes, coords, exe_path, optimize=True, forces=False):
    os.makedirs(dirname)
    fname = os.path.join(dirname,'run.sh')
    fout = open(fname, "w")
    txt=\
'''#!/bin/bash -l
export MOPAC_LICENSE={}
export LD_LIBRARY_PATH=$MOPAC_LICENSE:$LD_LIBRARY_PATH
$MOPAC_LICENSE/MOPAC2016.exe input.mop > log 2>&1
'''.format(exe_path)
    fout.write(txt)
    fout.close()
    st = os.stat(fname)
    os.chmod(fname, st.st_mode | stat.S_IEXEC)
    mopac_write_input(os.path.join(dirname,'input.mop'),atomtypes,coords,optimize, forces=forces)

if __name__ == '__main__':
    import time
    elements = ['H,'O',C','N']
    coords = [[0,0,0]]
    for e in elements:
        base_dir = os.path.join('mopac_atomrefs',e) 
        mopac_write_rundir(base_dir, [e], coords,'/home/gianni/QM/MOPAC2016')
        time.sleep(0.2)
        os.system(f'cd {base_dir};./run.sh')
        elem, _, energy = mopac_read_arc(os.path.join(base_dir,'input.arc'))
        print(f'{e}: {energy}')