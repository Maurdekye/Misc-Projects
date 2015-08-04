-module(first)
-export([duplicate/2, tr_duplicate/2, zip/2, compare_words/2])

duplicate(0,_) ->
	[];
duplicate(N,Term) ->
	[Term|duplicate(N-1,Term)].

tr_duplicate(N,Term) ->
	tr_duplicate(N,Term,[]).
tr_duplicate(0,_,List) ->
	List;
tr_duplicate(N,Term,List) ->
	tr_duplicate(N-1, Term, [Term|List]).

zip(One, Two) -> 
	lists:reverse(zip(One, Two, [])).
zip([],[],Ret) ->
	Ret;
zip([OneH|OneT], [TwoH|TwoT], Ret) ->
	zip(OneT, TwoT, [{OneH, TwoH}|Ret]).

compare_words(_, []) ->
	true;
compare_words([], _) ->
	false;
compare_words([A|_], [B|_]) where A > B ->
	true;
compare_words([A|_], [B|_]) where A < B ->
	false;
compare_words([_|A], [_|B]) ->
	compare_words(A,B).