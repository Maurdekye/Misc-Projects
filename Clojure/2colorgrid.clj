(defn rnumrow
  ([size]
    (rnumrow size []))
  ([size row]
    (if (zero? size)
      row
      (rnumrow (dec size) (conj row (+ (rand-int 15) 10))))))

(defn rnumgrid
  ([size]
   (rnumgrid size [] size))
  ([size grid progress]
   (if (zero? progress)
     grid
     (rnumgrid size (conj grid (rnumrow size)) (dec progress)))))

(defn getat [grid x y]
  (nth (nth grid y) x))

(defn setat [grid x y val]
  (assoc grid y (assoc (nth grid y) x val)))

(defn width [grid]
  (count (first grid)))

(defn height [grid]
  (count grid))

(defn pgrid
  ([grid]
   (pgrid grid 0 0))
  ([grid x y]
   (cond
     (= x (width grid))
       (do
         (println)
         (pgrid grid 0 (inc y)))
     (= y (height grid))
       (println)
     :else
       (do
         (print (str (getat grid x y) " "))
         (pgrid grid (inc x) y)))))

(defn cyclerow [row]
  (cons (last row) (butlast row)))

(defn shiftrow [grid y]
  (assoc grid y (cyclerow (nth grid y))))

(defn comparerows
  ([grid row1 row2]
   (comparerows grid row1 row2 0))
  ([grid row1 row2 col]
   (cond
     (= col (width grid))
       false
     (= (getat grid col row1) (getat grid col row2))
       true
     :else
       (comparerows grid row1 row2 (inc col)))))

(defn examinerow
  ([grid row]
   (examinerow grid row 0))
  ([grid row otherrow]
   (cond
     (= otherrow (height grid))
       false
     (= row otherrow)
       (examinerow grid row (inc otherrow))
     (comparerows grid row otherrow)
       true
     :else
       (examinerow grid row (inc otherrow)))))

(defn examinegrid
  ([grid]
   (examinegrid grid 0))
  ([grid row]
   (cond
     (= row (height grid))
       [grid 0 true]
     (examinerow grid row)
       [(shiftrow grid row) row false]
     :else
       (examinegrid grid (inc row)))))

(defn solve
  ([grid]
   (solve grid -1 0))
  ([grid lastrow shiftcount]
   (let [[newgrid shiftedrow done] (examinegrid grid)]
     (cond
       done
         [newgrid true]
       (= shiftcount (width grid))
         [newgrid false]
       :else
         (do
           (if (= shiftedrow lastrow)
             (solve newgrid shiftedrow (inc shiftcount))
             (solve newgrid shiftedrow 1)))))))

(def gr (rnumgrid 5))
(println "New Run")
(pgrid gr)
(let [[resultgrid couldsolve] (solve gr)]
  (if couldsolve
    (pgrid resultgrid)
    (println "Could not find solution")))
