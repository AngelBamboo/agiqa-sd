"""Source-disjoint benchmark harness for AI-generated image quality assessment.

The package keeps three things separate, because the paper's argument depends
on the distinction:

* ``datasets``  -- what the images and labels are, with a stable item order;
* ``protocols`` -- which images a model may train on and which it is scored on;
* ``runner``    -- how a model is scored and how the score is decomposed into
  pooled, within-generator and between-generator components.

Nothing here trains a model. Trained baselines keep their own scripts; this
package is the common yardstick that every model, trained or not, is measured
against.
"""
