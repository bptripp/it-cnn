l = 20;
rotate([45,45,45]) {
    rotate([90,0,0]) {
        cylinder(l,1,[0,0,0], $fn=50);
    }
    translate([0,-l,0]) {
        rotate([0,90,0]) {
            cylinder(l,1,[0,0,0], $fn=50);
        }
    }
    translate([l,0,0]) {
        rotate([90,0,0]) {
            cylinder(l,1,[0,0,0], $fn=50);
        }
    }
    translate([l,0,0]) {
        rotate([0,0,0]) {
            cylinder(l,1,[0,0,0], $fn=50);
        }
    }
    translate([l,0,l]) {
        rotate([0,-135,45]) {
            cylinder(l,1,[0,0,0], $fn=50);
        }
    }
    translate([l/2,-l/2,l*(1-.7071)]) {
        rotate([100,0,45]) {
            cylinder(l,1,[0,0,0], $fn=50);
        }
    }
    translate([l*1.19,-l*1.19,l*.12]) {
        rotate([0,90,45]) {
            cylinder(l,1,[0,0,0], $fn=50);
        }
    }
    translate([l*1.89,-l*.48,l*.12]) {
        rotate([0,90,135]) {
            cylinder(l,1,[0,0,0], $fn=50);
        }
    }
}