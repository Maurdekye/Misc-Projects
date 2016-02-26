nking = false; 
sking = false; 
_ = [] spawn {
    if (nking && sking) then {
        (units army_alpha) join (group northking);
        (units army_bravo) join (group southking);
    } else {
        if (nking) then {
            (units army_alpha) join (group northking);
            (units army_bravo) join (group northking);
        } else {
            if (sking) then {
                (units army_alpha) join (group southking);
                (units army_bravo) join (group southking);
            } else {
                (units army_alpha) join groundgroup;
                (units army_bravo) join groundgroup;
            };
        };
    };
};