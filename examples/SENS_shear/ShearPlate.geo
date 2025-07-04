//+
SetFactory("OpenCASCADE");
//+
w = 1.0;     //width of specimen
//+
h = 1.0;     //High of specimen
//+
w1 = 0.01;   //mesh refine region width
//+
dx = 0.04;   //mesh size
//+
dxr = 0.002; // refined mesh size

//+
Point(1) = {-0.5,   -0.5,   0.0,    dx};    //left-bottom point
//+
Point(2) = {-0.5+w, -0.5,   0.0,    dx};    //right-bottom point
//+
Point(3) = {-0.5+w, -0.5+h, 0.0,    dx};    //right-top point
//+
Point(4) = {-0.5,   -0.5+h, 0.0,    dx};    //left-top point


//+
Point(5) = {-0.5,    0.001,   0.0,    dx};
//+
Point(6) = { 0.0,    0.001,   0.0,    dxr/10};
//+
Point(7) = { 0.0,    -0.001,   0.0,    dxr/10};
//+
Point(8) = {-0.5,    -0.001,   0.0,    dx};

//+
Point(9) = { 0.04, -0.1, 0.0,    dxr};
//+
Point(10) = { 0.08, -0.18,  0.0,    dxr};
//+
Point(11) = { 0.2,  -0.325,   0.0,    dxr};
//+
Point(12)= { 0.35,  -0.43,  0.0,    dxr};


//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 1};
//+
BSpline(9) = {7, 9, 10, 11, 12, 2};
//+
Curve Loop(1) = {2, 3, 4, 5, 6, 7, 8, 1};
//+
Plane Surface(1) = {1};
//+
Physical Point("bottom_left", 10) = {1};
//+
Physical Point("bottom_right", 11) = {2};
//+
Physical Point("top_right", 12) = {3};
//+
Physical Point("top_left", 13) = {4};
//+
Physical Curve("left_edge", 14) = {4, 5, 7, 6, 8};
//+
Physical Curve("bottom_edge", 15) = {1};
//+
Physical Curve("right_edge", 16) = {2};
//+
Physical Curve("top_edge", 17) = {3};
//+
Physical Surface("block", 18) = {1};

//+
Field[1] = Distance;
//+
Field[1].EdgesList = {9};
//+
Field[2] = Threshold;
//+
Field[2].InField = 1;        // 与 Distance size field 关联
//+
Field[2].SizeMin = dxr;      // 最小网格尺寸
//+
Field[2].SizeMax = dx;       // 最大网格尺寸
//+
Field[2].DistMin = 0.075;      // 距离下限，低于此值时应用 SizeMin
//+
Field[2].DistMax =0.3;         // 距离上限，高于此值时应用 SizeMax

//+
Background Field = 2;
