#!/usr/bin/env python3
"""Convert ASP-H100 coarse dumps (dumps/asph_t*.bin) into a self-contained
HTML visualization (sph_visualization.html).

Dump format (version 3, magic 0x41535048):
  header: <2I10i5f> magic, version, nx,ny,nz,cx,cy,cz,sx,sy,sz,nfields,
          t_sim, dt, lx, ly, lz
  data  : cx*cy*cz coarse cells x 6 floats:
          [theta, qv, ql, u, v, w]  (cell averages; 0 => empty cell)
          cell index: idx = cxx + cx*(cyy + cy*czz)

Usage: python3 tools/make_viz.py [dumps_dir]
"""
import struct, glob, json, sys, os

def load(fn):
    d = open(fn, 'rb').read()
    h = struct.unpack('<2I10i5f', d[:68])
    magic, _, nx, ny, nz, cx, cy, cz, _, _, _, nf, t, dt, lx, ly, lz = h
    assert magic == 0x41535048, f"bad magic in {fn}"
    vals = struct.unpack(f'<{cx*cy*cz*nf}f', d[68:])
    def field(c):
        f = []
        for k in range(cz):
            sl = []
            for j in range(cy):
                row = []
                for i in range(cx):
                    idx = (k*cy + j)*cx + i
                    row.append(vals[idx*6 + c])
                sl.append(row)
            f.append(sl)
        return f
    return dict(t=t, lx=lx, ly=ly, lz=lz, nx=nx, ny=ny, nz=nz,
                cx=cx, cy=cy, cz=cz,
                fields={name: field(i) for i, name in
                        enumerate(['theta', 'qv', 'ql', 'u', 'v', 'w'])})

def main():
    ddir = sys.argv[1] if len(sys.argv) > 1 else 'dumps'
    files = sorted(glob.glob(os.path.join(ddir, 'asph_t*.bin')))
    assert files, f"no dumps found in {ddir}"
    frames = [load(f) for f in files]
    t0 = frames[0]
    theta0 = t0['fields']['theta'][0][0][0]
    data = dict(
        Lx=t0['lx'], Ly=t0['ly'], Lz=t0['lz'],
        cx=t0['cx'], cy=t0['cy'], cz=t0['cz'],
        dx=t0['lx']/t0['nx'], dy=t0['ly']/t0['ny'], dz=t0['lz']/t0['nz'],
        theta_ref=round(theta0, 2),
        times=[round(fr['t'], 1) for fr in frames],
        theta=[[[[round(v, 2) for v in row] for row in sl] for sl in fr['fields']['theta']] for fr in frames],
        qv=[[[[round(v, 5) for v in row] for row in sl] for sl in fr['fields']['qv']] for fr in frames],
        ql=[[[[round(v, 5) for v in row] for row in sl] for sl in fr['fields']['ql']] for fr in frames],
        w=[[[[round(v, 3) for v in row] for row in sl] for sl in fr['fields']['w']] for fr in frames],
    )
    print("frames:", data['times'])
    html = build(data)
    with open('sph_visualization.html', 'w') as f:
        f.write(html)
    print("wrote sph_visualization.html:", os.path.getsize('sph_visualization.html')//1024, "KB")

def build(data):
    here = os.path.dirname(__file__)
    tmpl = open(os.path.join(here, 'template.html')).read()
    n = len(data['times'])
    return (tmpl.replace('__DATA__', json.dumps(data, separators=(',', ':')))
                .replace('__NMAX__', str(n - 1))
                .replace('__ZMAX__', str(data['cz'] - 1))
                .replace('__ZDEF__', str(data['cz'] // 2)))

if __name__ == '__main__':
    main()
