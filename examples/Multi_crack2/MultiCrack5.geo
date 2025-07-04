//+
SetFactory("OpenCASCADE");
//+
w = 50.0;     //width of specimen
//+
h = 50.0;     //High of specimen
//+
dx = 1.0;   //mesh size
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
Point(5) = {19.5-1-w/2,     15-h/2,   0.0,    dx};
//+
Point(6) = {19.5-w/2,       15+1-h/2,   0.0,    dx};
//+
Point(7) = {19.5+4-3-w/2,   15-h/2,   0.0,    dx};
//+
Point(8) = {19.5+4-w/2,     15 + 3 - h/2,   0.0,    dx};
//+
Point(9) = {19.5+4+8-5-w/2, 15-h/2, 0.0,   dx};
//+
Point(10) = {19.5+4+8-w/2,  15 + 5 - h/2,0.0,    dx};

//+
Point(11) = {19.5-1-w/2,    15-h/2+10,   0.0,    dx};
//+
Point(12) = {19.5-w/2,      15+1-h/2+10,   0.0,    dx};
//+
Point(13) = {19.5+4-3-w/2,  15-h/2+10,   0.0,    dx};
//+
Point(14) = {19.5+4-w/2,    15 + 3 - h/2+10,   0.0,    dx};
//+
Point(15) = {19.5+4+8-5-w/2,15-h/2+10, 0.0,   dx};
//+
Point(16) = {19.5+4+8-w/2,  15 + 5 - h/2+10,0.0,    dx};

//+
Point(17) = {19.5-1-w/2, 15-h/2+20,   0.0,    dx};
//+
Point(18) = {19.5-w/2, 15+1-h/2+20,   0.0,    dx};
//+
Point(19) = {19.5+4-3-w/2,  15-h/2+20,   0.0,    dx};
//+
Point(20) = {19.5+4-w/2,  15 + 3 - h/2+20,   0.0,    dx};
//+
Point(21) = {19.5+4+8-5-w/2,     15-h/2+20, 0.0,   dx};
//+
Point(22) = {19.5+4+8-w/2,   15 + 5 - h/2+20,0.0,    dx};

//+
Point(23) = {-w/2, -10, 0.0,   dx};
//+
Point(24) = { w/2,  -5, 0.0,    dx};
//+
Point(25) = {2.5,   -8.5, 0.0,   dx};
//+
Point(26) = {-2.5,  -8.5,0.0,    dx};

//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {18, 17};
//+
Line(6) = {19, 20};
//+
Line(7) = {21, 22};
//+
Line(8) = {9, 10};
//+
Line(9) = {8, 7};
//+
Line(10) = {6, 5};
//+
Line(11) = {11, 12};
//+
Line(12) = {14, 13};
//+
Line(13) = {15, 16};
//+
Line(14) = {24, 10};
//+
Line(15) = {5, 23};
//+
Line(16) = {6, 7};
//+
Line(17) = {26, 9};
//+
Line(18) = {25, 8};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};

//+
Physical Point("bottom_left", 19) = {1};
//+
Physical Point("bottom_right", 20) = {2};
//+
Physical Point("top_right", 21) = {3};
//+
Physical Point("top_left", 22) = {4};
//+
Physical Curve("left_edge", 23) = {1};
//+
Physical Curve("bottom_edge", 24) = {2};
//+
Physical Curve("right_edge", 25) = {3};
//+
Physical Curve("top_edge", 26) = {4};
//+
Physical Surface("block", 27) = {1};

//+
Field[1] = Distance;
//+
Field[1].EdgesList = {8,9,10,14,15,16,17,18};
//+
Field[2] = Threshold;
//+
Field[2].InField = 1;        // 与 Distance size field 关联
//+
Field[2].SizeMin = dxr;      // 最小网格尺寸
//+
Field[2].SizeMax = dx;       // 最大网格尺寸
//+
Field[2].DistMin = 5;     // 距离下限，低于此值时应用 SizeMin
//+
Field[2].DistMax = 30;      // 距离上限，高于此值时应用 SizeMax

//+
Field[3] = Distance;
//+
Field[3].EdgesList = {5,6,7,11,12,13};
//+
Field[4] = Threshold;
//+
Field[4].InField = 3;        // 与 Distance size field 关联
//+
Field[4].SizeMin = dxr;      // 最小网格尺寸
//+
Field[4].SizeMax = dx;       // 最大网格尺寸
//+
Field[4].DistMin = 5;     // 距离下限，低于此值时应用 SizeMin
//+
Field[4].DistMax = 30;      // 距离上限，高于此值时应用 SizeMax

//+
Field[5] = Min;
//+
Field[5].FieldsList = {2,4};
//+
Background Field = 5;

