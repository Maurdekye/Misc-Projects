#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.  ;
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.
#IfWinActive ahk_class grcWindow ; Disables hotkeys when alt-tabbed or GTA is closed.

SetKeyDelay, [10, 10]

; Auto-Scarf
*Numpad1::
  Send, {m}{Down}{Down}{Enter}{Enter}{Up}{Up}{Right}{m}
  return

; Armor
*Numpad7::
  Send, {m}{Down}{Enter}{Down}{Enter}{Up}{Up}{Up}{Enter}{m}
  return

; Light Snacks
*Numpad9::
  Send, {m}{Down}{Enter}{Down}{Down}{Enter}{Enter}{Enter}{Enter}{Enter}{m}
  return

; Heavy Snacks
*Numpad6::
  Send, {m}{Down}{Enter}{Down}{Down}{Enter}{Down}{Enter}{m}
  return

; Accept Job
*Numpad8::
  Send, {Up}
  Sleep, 10
  Send, {Enter}
  Sleep, 10
  Send, {Enter}
  Sleep, 10
  Send, {Enter}
  Sleep, 10
  Send, {Enter}
  return

; Auto-Mechanic
*Numpad0::
  Send, {Up}
  Sleep, 600
  Send, {Right}
  Sleep, 50
  Send, {Up}{Enter}
  Sleep, 50
  Send, {Left}
  Sleep, 150
  Send, {Up}
  Sleep, 10
  Send, {Enter}
  return

; Script Tester
*Numpad5::
  Send, tactive
  Sleep, 300
  Send, {Esc}
  return

; What the fuck did you just fucking say about me, you little bitch? I'll have you know I graduated top of my class in the Navy Seals, and
; I've been involved in numerous secret raids on Al-Quaeda, and I have over 300 confirmed kills. I am trained in gorilla warfare and I'm
; the top sniper in the entire US armed forces. You are nothing to me but just another target. I will wipe you the fuck out with precision
; the likes of which has never been seen before on this Earth, mark my fucking words. You think you can get away with saying that shit to
; me over the Internet? Think again, fucker. As we speak I am contacting my secret network of spies across the USA and your IP is being
; traced right now so you better prepare for the storm, maggot. The storm that wipes out the pathetic little thing you call your life.
; You're fucking dead, kid. I can be anywhere, anytime, and I can kill you in over seven hundred ways, and that's just with my bare hands.
; Not only am I extensively trained in unarmed combat, but I have access to the entire arsenal of the United States Marine Corps and I
; will use it to its full extent to wipe your miserable ass off the face of the continent, you little shit. If only you could have known
; what unholy retribution your little "clever" comment was about to bring down upon you, maybe you would have held your fucking tongue.
; But you couldn't, you didn't, and now you're paying the price, you goddamn idiot. I will shit fury all over you and you will drown in 
; it. You're fucking dead, kiddo.
NumpadDot::
  return
  Send, t
  Sleep, 50
  Send, What the fuck did you just fucking say about me, you little bitch? I'll have you know I graduated top of my class in the Navy Seals, and
  Send, {Enter}
  Send, t
  Sleep, 50
  Send, I've been involved in numerous secret raids on Al-Quaeda, and I have over 300 confirmed kills. I am trained in gorilla warfare and I'm
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, the top sniper in the entire US armed forces. You are nothing to me but just another target. I will wipe you the fuck out with precision
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, the likes of which has never been seen before on this Earth, mark my fucking words. You think you can get away with saying that shit to
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, me over the Internet? Think again, fucker. As we speak I am contacting my secret network of spies across the USA and your IP is being
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, traced right now so you better prepare for the storm, maggot. The storm that wipes out the pathetic little thing you call your life.
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, You're fucking dead, kid. I can be anywhere, anytime, and I can kill you in over seven hundred ways, and that's just with my bare hands.
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, Not only am I extensively trained in unarmed combat, but I have access to the entire arsenal of the United States Marine Corps and I
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, will use it to its full extent to wipe your miserable ass off the face of the continent, you little shit. If only you could have known
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, what unholy retribution your little "clever" comment was about to bring down upon you, maybe you would have held your fucking tongue.
  Send, {Enter}
  Sleep, 10
  Send, t
  Sleep, 50
  Send, But you couldn't, you didn't, and now you're paying the price, you goddamn idiot. I will shit fury all over you and you will drown in
  Send, {Enter}
  Sleep, 2000
  Send, t
  Sleep, 50
  Send, it. You're fucking dead, kiddo.
  Send, {Enter}
  return

; Reload Key
*NumpadAdd::Reload