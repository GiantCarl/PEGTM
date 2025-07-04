// Gmsh project created on Sat Sep 21 08:45:24 2024
SetFactory("OpenCASCADE");
//+
dx = 0.04 ;//mesh size
//+
dx1= 0.002;//refine mesh size
//+
Point(1) = {-0.5,   -0.5,   0.0,    dx};    //left-bottom point
//+
Point(2) = {0, -0.5,   0.0,    dx};         //middle-bottom point
//+
Point(3) = {0, 0, 0.0,    dx1};             //origin point
//+
Point(4) = {0.44, 0, 0.0,    dx/4};           //load point
//+
Point(5) = {0.5,   0.0, 0.0,    dx};        //middle-left point
//+
Point(6) = {0.5,  0.5,   0.0,    dx};      // right-top point
//+
Point(7) = {-0.5, 0.5,   0.0,    dx};      // left-top point
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
Line(7) = {7, 1};
//+
Curve Loop(1) = {7, 1, 2, 3, 4, 5, 6};
//+
Plane Surface(1) = {1};
//+
Transfinite Curve {4} = 6 Using Progression 1;
//+
Physical Curve("left_edge", 8) = {7};
//+
Physical Curve("bottom_edge", 9) = {1};
//+
Physical Curve("right_edge", 10) = {5};
//+
Physical Curve("top_edge", 11) = {6};
//+
Physical Curve("load_edge", 12) = {4};
//+
Physical Point("bottom_left", 13) = {1};
//+
Physical Point("bottom_right", 14) = {2};
//+
Physical Point("top_right", 15) = {6};
//+
Physical Point("top_left", 16) = {7};
//+
Physical Surface("block", 17) = {1};
//+
Field[1] = Ball;
//+
Field[1].Radius = 0.1;
//+
Field[1].Thickness = 0.1;
//+
Field[1].VIn = dx1;
//+
Field[1].VOut = dx;
//+
Field[1].XCenter = -0.05;
//+
Field[1].YCenter = 0.05;
//+
Field[2] = Box;
//+
Field[2].Thickness = 0.1;
//+
Field[2].VIn = dx1;
//+
Field[2].VOut = dx;
//+
Field[2].XMax = -0.05;
//+
Field[2].XMin = -0.5;
//+
Field[2].YMax = 0.15;
//+
Field[2].YMin = -0.05;
//+
Field[3] = Min;
//+
Field[3].FieldsList = {1, 2};
//+
Background Field = 3;


