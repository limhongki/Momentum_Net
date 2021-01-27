
nx = 420;
sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd');
sg.na = 123; %number of projection views
down = 1;
ig = image_geom('nx', nx, 'dx', 500/512, 'down', down);
ig.mask = ig.circ > 0;
A = Gtomo2_dscmex(sg, ig);  %projection matrix

