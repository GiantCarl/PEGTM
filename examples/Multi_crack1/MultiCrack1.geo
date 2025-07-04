//+
SetFactory("OpenCASCADE");
//+
w = 50.0;     //width of specimen
//+
h = 50.0;     //High of specimen
//+
dx = 1;   //mesh size
//+
dxr = 0.1; // refined mesh size

//+
Point(1) = {-w/2, -h/2,   0.0,    dx};    //left-bottom point
//+
Point(2) = { w/2, -h/2,   0.0,    dx};    //right-bottom point
//+
Point(3) = { w/2,  h/2,   0.0,    dx};    //right-top point
//+
Point(4) = {-w/2,  h/2,   0.0,    dx};    //left-top point

//+
Point(5) = {12.5-w/2, 17.5-h/2,   0.0,    dx};
//+
Point(6) = {12.5 + 5 -w/2, 17.5 + 5 -h/2,    0.0,    dx};
//+
Point(7) = {12.5 + 10 -w/2, 17.5 + 5 -h/2,   0.0,    dx};
//+
Point(8) = {12.5 + 15 -w/2, 17.5 + 10 -h/2,   0.0,    dx};
//+
Point(9) = {12.5 + 20 -w/2, 17.5 + 10 -h/2,   0.0,    dx};
//+
Point(10) = {12.5 + 25 -w/2, 17.5 + 15 -h/2,   0.0,    dx};
//+
Point(11) = {-w/2, 17.5-h/2,   0.0,    dx};
//+
Point(12) = {w/2, 17.5 + 15 -h/2,   0.0,    dx};

//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {5, 6};
//+
Line(6) = {7, 8};
//+
Line(7) = {9, 10};
//+
Line(8) = {5, 11};
//+
Line(9) = {6, 7};
//+
Line(10) = {8, 9};
//+
Line(11) = {10, 12};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};

//+
Physical Point("bottom_left", 13) = {1};
//+
Physical Point("bottom_right", 14) = {2};
//+
Physical Point("top_right", 15) = {3};
//+
Physical Point("top_left", 16) = {4};
//+
Physical Curve("left_edge", 17) = {1};
//+
Physical Curve("bottom_edge", 18) = { 2};
//+
Physical Curve("right_edge", 19) = {3};
//+
Physical Curve("top_edge", 20) = {4};
//+
Physical Surface("block", 21) = {1};

//+
Field[1] = Distance;
//+
Field[1].EdgesList = {5,6,7,8,9,10,11};
//+
Field[2] = Threshold;
//+
Field[2].InField = 1;        // 与 Distance size field 关联
//+
Field[2].SizeMin = dxr;      // 最小网格尺寸
//+
Field[2].SizeMax = dx;       // 最大网格尺寸
//+
Field[2].DistMin = 3;      // 距离下限，低于此值时应用 SizeMin
//+
Field[2].DistMax =20;         // 距离上限，高于此值时应用 SizeMax


//+
Background Field = 2;

