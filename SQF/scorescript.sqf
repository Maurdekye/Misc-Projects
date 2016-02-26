weight = 0;
wingoal = 100;
_ = [] spawn {
    loophook = true;
    while {loophook} do {
        bluscore = {(side _x == west) and (_x inArea contention)} count allUnits;
        opscore = {(side _x == east) and (_x inArea contention)} count allUnits;
        unittotal = bluscore + opscore;
        momentum = 0;
        if (unittotal > 0) {
            if (bluscore == 0) {
                momentum = -1;
            } else{if (opscore == 0) {
                momentum = 1;
            } else {
                bluepercent = bluescore / unittotal;
                oppercent = opscore / unittotal;
                momentum = bluepercent - oppercent
            };};
        } else {};
        weight = weight + momentum;
        hintstring = (str weight) + "/" + (str wingoal) + " : " + (str momentum);
        if (weight > wingoal) {
            loophook = false;
            titleText ["Success!", "black out", 8];
            _ = [] spawn {sleep 9; endMission "end1";};
        } else{if (weight < -wingoal) {
            loophook = false;
            titleText ["Failure.", "black out", 8];
            _ = [] spawn {sleep 9; endMission "loser";};
        } else {};};
    };
};