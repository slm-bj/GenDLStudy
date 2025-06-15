#lang racket

(require plot)

; Helper function to generate normal random numbers
(define (random-normal [mu 0.0] [sigma 1.0])
  (let* ([u1 (random)]
         [u2 (random)]
         [z (sqrt (* -2.0 (log u1)))])
    (+ mu (* sigma (* z (cos (* 2.0 pi u2)))))))

; Define the bandit problem
(define (make-bandit arms)
  (let ([means (for/list ([i (in-range arms)])
                 (random-normal))])
    (λ (action)
      (let ([best-action (argmax (λ (i) (list-ref means i)) (range arms))])
        (values (random-normal (list-ref means action) 1) best-action)))))

; Define the agent
(define (make-agent arms epsilon)
  (let ([Q (make-vector arms 0.0)]
        [N (make-vector arms 0)]
        [epsilon epsilon])
    (λ (action reward)
      (vector-set! N action (+ (vector-ref N action) 1))
      (vector-set! Q action (+ (vector-ref Q action)
                               (/ (- reward (vector-ref Q action))
                                  (vector-ref N action)))))))

(define (choose-action agent arms epsilon)
  (if (< (random) epsilon)
      (random arms)
      (argmax (λ (i) (vector-ref (car agent) i)) (range arms))))

; Run the experiment
(define (run-experiment bandit agent arms trials epsilon)
  (let ([rewards (make-vector trials 0.0)]
        [best-action-counts (make-vector trials 0)])
    (for ([t (in-range trials)])
      (let-values ([(reward best-action) (bandit (choose-action agent arms epsilon))])
        (vector-set! rewards t reward)
        (vector-set! best-action-counts t (= (choose-action agent arms epsilon) best-action))
        ((cdr agent) (choose-action agent arms epsilon) reward)))
    (values rewards best-action-counts)))

; Plot the results
(define (plot-results rewards best-action-counts epsilon)
  (plot (list (lines (build-list (vector-length rewards) values) rewards
                     #:label (format "Epsilon: ~a" epsilon)
                     #:color 'blue)
              (lines (build-list (vector-length best-action-counts) values) best-action-counts
                     #:label (format "Epsilon: ~a" epsilon)
                     #:color 'red))
        #:x-label "Trials"
        #:y-label "Average Reward"
        #:title "Experiment Results"))

(define (main)
  (let* ([arms 10]
         [trials 1000]
         [epsilon 0.1]
         [bandit (make-bandit arms)]
         [agent (cons (make-vector arms 0.0) (make-agent arms epsilon))]
         [rewards (vector-copy (car (run-experiment bandit agent arms trials epsilon)))]
         [best-action-counts (vector-copy (cdr (run-experiment bandit agent arms trials epsilon)))])
    (plot-results rewards best-action-counts epsilon)))

(main)
