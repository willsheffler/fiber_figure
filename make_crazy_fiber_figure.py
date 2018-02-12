import pymol
from collections import namedtuple
from pymol import cmd
import sys

import numpy as np
import homog as hm
import fire


numline = 0
numseg = 0


def cgo_cyl(c1, c2, r, col=(1, 1, 1), col2=None):
    from pymol import cgo
    if not col2:
        col2 = col
    return [
        cgo.CYLINDER, c1[0], c1[1], c1[2],
        c2[0], c2[1], c2[2], r,
        col[0], col[1], col[2], col2[0], col2[1], col2[2], ]


def showcyl(c1, c2, r, col=(1, 1, 1), col2=None, lbl=''):
    if not lbl:
        global numseg
        lbl = "seg%i" % numseg
        numseg += 1
    cmd.delete(lbl)
    v = cmd.get_view()
    cmd.load_cgo(cgo_cyl(c1=c1, c2=c2, r=r, col=col, col2=col2), lbl)
    cmd.set_view(v)


def calc_com(sel="all", state=1):
    # assumes equal weights (best called with "and name ca" suffix)
    model = cmd.get_model(sel, state)
    c = np.zeros(3)
    for a in model.atom:
        c += np.array(a.coord)
    c = c / len(model.atom)
    return c


def xform_between_chains(chain1, chain2, sele='name CA', state=1):
    assert chain1 != chain2
    atoms1 = cmd.get_model(f'chain {chain1} and ({sele})', state).atom
    atoms2 = cmd.get_model(f'chain {chain2} and ({sele})', state).atom
    assert len(atoms1) > 2
    assert len(atoms1) == len(atoms2)
    atoms1 = atoms1[0], atoms1[int(len(atoms1) / 2)], atoms1[-1]
    atoms2 = atoms2[0], atoms2[int(len(atoms2) / 2)], atoms2[-1]
    crds1 = [a.coord for a in atoms1]
    crds2 = [a.coord for a in atoms2]
    stub1 = hm.hstub(*crds1)
    stub2 = hm.hstub(*crds2)
    xform = stub2 @ hm.hinv(stub1)
    return xform


def helix_axis(sele='all', state=1):
    xform = xform_between_chains('A', 'B', sele=f'({sele}) and name CA',
                                 state=state)
    axis, ang, cen = hm.axis_ang_cen_of(xform)
    return axis, cen


def align_helix_axis(sele='all', state=1):
    print('align_helix_axis', f'({sele}) and name CA')
    axis, cen = helix_axis(sele, state)
    print('axis', axis)
    print('cen ', cen)
    # showline(axis * 1000, cen)
    cmd.rotate(list((axis[:3] + [0, 0, 1]) / 2), 180, selection=sele)
    cmd.translate(list(-calc_com(f'{sele} and name CA')))
    axis, cen = helix_axis(sele, state)
    cen[2] = 0
    cmd.translate(list(-cen))

Unit = namedtuple('Unit', 'z sele com chain stub'.split())


def helix_units(sele='all', state=1):
    print('helix_units')
    atoms = cmd.get_model(f'{sele} and chain A', state).atom
    unit_size = atoms[-1].index - atoms[0].index + 1
    num_units = cmd.count_atoms(sele) / unit_size
    assert int(num_units) == num_units
    assert atoms[0].chain == atoms[-1].chain
    seles = []
    for i in range(int(num_units)):
        start, stop = i * unit_size + 1, (i + 1) * unit_size
        sel = f'(index {start}-{stop})'
        com = calc_com(f'{sel} and name CA')
        a = cmd.get_model(f'{sel} and name CA').atom
        stub = hm.hstub(a[0].coord,
                        a[int(len(a) / 2)].coord,
                        a[-1].coord)
        seles.append(Unit(com[2], sel, com, a[0].chain, stub))
    return seles


def primary_xform_commutator(units, state=1, **kw):
    print('stub 0:')
    print(units[0].stub)
    closest = None
    for j in range(1, 20):
        x = hm.hinv(units[0].stub) @ units[j].stub
        # print(f'stub {j}:')
        # print(units[j].stub)
        # print(x)
        axis, ang, cen = hm.axis_ang_cen_of(x)
        helical = hm.hdot(axis, x[:, 3])
        print(j, helical, x[2, 3])
        if abs(helical) > 0.01:
            return x
    return None
    # assert 0, 'couldn\'t find primary xform'


def sort_and_calc_neighbors(
        units, sele='all', state=1, window=6, discut=8, **kw):
    print('sort_and_calc_neighbors')
    units.sort()
    xprimary = primary_xform_commutator(units, state=state)
    if xprimary is not None:
        primary_neighbor = {}
        other_neighbor = {}
    else:
        neighbors = []
    for i in range(len(units)):
        isele = units[i].sele
        for j in range(i + 1, min(i + window + 1, len(units))):
            # if i == j: continue
            xcomm = hm.hinv(units[i].stub) @ units[j].stub
            jsele = units[j].sele
            isect_sele = (f'({isele} and name CA) within {discut} of ' +
                          f'({jsele} and name CA)')
            n = cmd.select(isect_sele, state=state)
            if n > 0:
                if xprimary is not None:
                    if np.allclose(xcomm, xprimary, atol=1e-2, rtol=1e-2):
                        assert i not in primary_neighbor
                        primary_neighbor[i] = j
                    else:
                        assert j not in other_neighbor
                        other_neighbor[i] = j
                else:
                    neighbors.append((i, j))
    if xprimary is not None:
        assert len(primary_neighbor) > max(0, len(units) - 10)
        assert len(other_neighbor) > max(0, len(units) - 10)
        return primary_neighbor, other_neighbor
    else:
        return neighbors, None


def show_interaction(u, v, primary, scale):
    u, v = np.asarray(u), np.asarray(v)
    radius = 4 if primary else 2
    col = (1, 0, 0) if primary else (0, 0, 1)
    d = v - u
    q = 1 / scale
    showcyl(u + d * q, u + d * (1 - q), radius, col=col)


def explode_helix(sele='all', scale=3, state=1, **kw):
    print('explode_helix: aligning to 0,0,0 / 0,0,1')
    align_helix_axis('original', state=state)
    unit_selections = list(helix_units(sele, state))
    primary_nbr, other_nbr = sort_and_calc_neighbors(unit_selections, **kw)
    coms = []
    for i, unit in enumerate(unit_selections):
        cmd.translate(list(unit.com * (scale - 1.0)), selection=unit.sele)
        coms.append(unit.com * scale)
    if other_nbr is not None:
        for i, j in primary_nbr.items():
            show_interaction(coms[i], coms[j], primary=True, scale=scale)
        for i, j in other_nbr.items():
            show_interaction(coms[i], coms[j], primary=False, scale=scale)
    else:
        for i, j in primary_nbr:
            show_interaction(coms[i], coms[j], primary=False, scale=scale)


def launch_pymol(**kw):
    pymol.pymol_argv = ['pymol']
    if 'headless' in kw and kw['headless']:
        pymol.pymol_argv = ['pymol', '-c']
    pymol.finish_launching()
    cmd.set('stop_on_exceptions', True)


def make_helix_graphics(pdbfile, scale=3, discut=8, window=8, **kw):
    print('make_helix_graphics', pdbfile, kw)
    cmd.delete('all')
    cmd.do('bg white')
    cmd.do('stereo off')
    cmd.load(pdbfile, 'original')
    cmd.remove('not name ca')
    cmd.hide('ev')
    cmd.show('sph')
    cmd.set('sphere_scale', '3')
    pymol.util.cbc()
    explode_helix('original', scale,
                  state=1, discut=discut, window=window)
    cmd.save(f'{pdbfile}_exploded.pse')


def main(*args, **kw):
    launch_pymol(**kw)
    for arg in args:
        make_helix_graphics(arg, **kw)

if __name__ == '__main__':
    # sys.argv.append('--headless=False')
    fire.Fire(main)
