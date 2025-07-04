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
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};

//+
Point(5) = {-0.5,   0.0,  0.0,    dx};    //Crack
//+
Point(6) = { 0.0,   0.0,  0.0,    dx};    //Crack
//+
Line(5) = {5, 6};

//+
Point(7) = { 0.04,   0.07,  0.0,    dx};    //Crack
//+
Point(8) = { 0.08,   0.15,  0.0,    dx};    //Crack
//+
Point(9) = { 0.15,   0.216,  0.0,    dx};    //Crack
//+
Point(10) = { 0.225,   0.27,  0.0,    dx};   //Crack
//+
Point(11) = { 0.3,   0.3,  0.0,    dx};   //Crack
//+
Spline(6) = {6,7, 8, 9, 10,11};

//+
Point(12) = { 0.05,   -0.076,  0.0,    dx};    //Crack
//+
Point(13) = { 0.08,  -0.128,  0.0,    dx};    //Crack
//+
Point(14) = { 0.15,  -0.198,  0.0,    dx};    //Crack
//+
Point(15) = { 0.225,  -0.255,  0.0,    dx};   //Crack
//+
Point(16) = { 0.3,  -0.3,  0.0,    dx};   //Crack
//+
Spline(7) = {6,12,13,14,15,16};


//+
Physical Curve("left_edge", 8) = {4};
//+
Physical Curve("bottom_edge", 9) = {1};
//+
Physical Curve("right_edge", 10) = {2};
//+
Physical Curve("top_edge", 11) = {3};
//+
Physical Surface("block", 12) = {1};

//+
Field[1] = Distance;
//+
Field[1].EdgesList = {5,6,7};
//+
Field[2] = Threshold;
//+
Field[2].InField = 1;        // 与 Distance size field 关联
//+
Field[2].SizeMin = dxr;      // 最小网格尺寸
//+
Field[2].SizeMax = dx;       // 最大网格尺寸
//+
Field[2].DistMin = 0.1;      // 距离下限，低于此值时应用 SizeMin
//+
Field[2].DistMax =0.3;         // 距离上限，高于此值时应用 SizeMax
//+
Background Field = 2;
