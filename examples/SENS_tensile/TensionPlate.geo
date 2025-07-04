// Gmsh project created on Sat Sep 21 08:45:24 2024
SetFactory("OpenCASCADE");

//+
w = 1.0;   //width of specimen
//+
h = 1.0;  //High of specimen
//+
w1 = 0.01; //mesh refine region width
//+
dx = 0.04; //mesh size
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
Physical Point("bottom_left", 5) = {1};
//+
Physical Point("bottom_right", 6) = {2};
//+
Physical Point("top_right", 7) = {3};
//+
Physical Point("top_left", 8) = {4};
//+
Physical Curve("left_edge", 9) = {4};
//+
Physical Curve("bottom_edge", 10) = {1};
//+
Physical Curve("right_edge", 11) = {2};
//+
Physical Curve("top_edge", 12) = {3};
//+
Physical Surface("block", 13) = {1};

//+
Field[1] = Box;
//+
Field[1].Thickness = 0.2;
//+
Field[1].VIn = dxr;
//+
Field[1].VOut = dx;
//+
Field[1].XMax = 0.5;
//+
Field[1].XMin = -0.5;
//+
Field[1].YMax = 0.05;
//+
Field[1].YMin = -0.05;

//+
Background Field = 1;
