_ = this addEventHandler ["Fired", {    
    ammoname = _this select 4;
    if (currentsong == -1) then {
        currentsong = 0;
    } else {
        if (ammoname == "B_556x45_Ball_Tracer_Green" || ammoname == "B_556x45_Ball") then {
            currentsong = (currentsong + 1) % (count songclasses);
        } else {
            currentsong = currentsong - 1;
            if (currentsong < 0) then {
                currentsong = (count songnames) - 1;
            } else {};
        };
    };
    playMusic "";
    playMusic (songclasses select currentsong);       
    if ((songnames select currentsong) == "") then {          
        hintSilent ("Class: " + (songclasses select currentsong));      
    } else {          
        hintSilent ("Track: " + (songnames select currentsong));      
    };
}];

"Track07_ActionDark" "Track09_Night_percussions" "Track10_StageB_action" "LeadTrack01_F_EPA" "EventTrack02_F_EPA" "EventTrack02_F_EPB" "BackgroundTrack01_F_EPB"