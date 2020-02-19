syms a1 a2 b1 b2 c1 c2;
assume(a1 > 0);
assume(a2 > 0);
assume(a1 * b1 - c1^2 == 1);
assume(a2 * b2 - c2^2 == 1);

A = [a1 c1; c1 b1];
B = [a2 c2; c2 b2];
spd_dist = sqrt(sum(log(eig(A\B)).^2));
spd_dist = simplify(spd_dist);

x = [(a1 + b1) / 2, (a1 - b1) / 2, c1];
y = [(a2 + b2) / 2, (a2 - b2) / 2, c2];
hyp_dist = acosh(x(1) * y(1) - x(2) * y(2) - x(3) * y(3));
hyp_dist = simplify(hyp_dist);

isequal(spd_dist, hyp_dist)
isequal(spd_dist, hyp_dist)
isequal(spd_dist, sqrt(2) * hyp_dist)  % Does not work as expected!

a1 = 5; b1 = 1; c1 = 2;
a2 = 2; b2 = 1; c2 = 1;
isequal(double(subs(spd_dist)), sqrt(2) * double(subs(hyp_dist)))
